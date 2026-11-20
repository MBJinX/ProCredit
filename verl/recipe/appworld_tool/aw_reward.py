"""Verified progress and rewards for AppWorld.

Progress is the fraction of initially failing checks (label no_op_fail)
that currently pass. Per-turn shaping is coef * (phi_t - phi_previous).
The keep terminal mode adds the outcome reward to the final progress delta;
the ground terminal mode instead resets terminal progress to zero.
ProCredit uses keep with coef=0.5. Scalar-only and ground modes remain
available for baseline and ablation comparisons.
"""

import hashlib
import os
from typing import Optional

NET_LABEL = "no_op_fail"


def net_counts(eval_dict: dict) -> tuple[int, int]:
    """返回 (net_pass, net_n)。"""
    passes = eval_dict.get("passes", []) or []
    failures = eval_dict.get("failures", []) or []
    net_pass = sum(1 for p in passes if p.get("label") == NET_LABEL)
    net_n = net_pass + sum(1 for f in failures if f.get("label") == NET_LABEL)
    return net_pass, net_n


def net_requirements(eval_dict: dict) -> tuple[list[str], list[str]]:
    """返回 (通过的净断言 requirement 列表, 全部净断言 requirement 列表)。

    AppWorld 的断言项是 TypedDict `Pass{requirement,label}` / `Failure{requirement,trace,label}`
    (appworld_dev/src/appworld/evaluator.py:40-48)——**没有数字 id,`requirement` 文本即唯一标识**。
    净断言 = label == "no_op_fail"(与 net_counts 同口径)。
    """
    passes = eval_dict.get("passes", []) or []
    failures = eval_dict.get("failures", []) or []
    passed = [p["requirement"] for p in passes if p.get("label") == NET_LABEL]
    failed = [f["requirement"] for f in failures if f.get("label") == NET_LABEL]
    return passed, passed + failed


def anchor_of(passed_requirements: list[str]) -> str:
    """进度等价类锚点 = 已通过净断言集合的稳定 hash(GiGPO anchor_obs 的替代)。

    AppWorld 的 obs 是代码执行输出全文,开放状态空间锚不住;改用执行反馈定义的进度集合。
    用 frozenset 语义(顺序无关、重复 requirement 去重)+ sha1 前 16 位。
    空集合(尚无断言通过)也是合法锚点——起手状态本就该互为对照。
    """
    key = "\x1f".join(sorted(set(passed_requirements)))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def phi_of(eval_dict: dict) -> float:
    np_, nn = net_counts(eval_dict)
    return np_ / nn if nn > 0 else 0.0


def shaping_enabled() -> bool:
    return os.environ.get("APPWORLD_SHAPING", "0") == "1"


def shaping_coef() -> float:
    return float(os.environ.get("APPWORLD_SHAPING_COEF", "0.5"))


def step_credit_enabled() -> bool:
    return os.environ.get("APPWORLD_STEP_CREDIT", "0") == "1"


def step_terminal_mode() -> str:
    """返回步级终态口径：ground=终态归零，keep=保留最终进度。"""
    mode = os.environ.get("APPWORLD_STEP_TERMINAL", "ground")
    if mode not in ("ground", "keep"):
        raise ValueError(f"未知 APPWORLD_STEP_TERMINAL: {mode}")
    return mode


def compute_reward(outcome: float, phi_seq: list[float]) -> dict:
    """phi_seq = [Φ_0(初始 evaluate), Φ_1, ..., Φ_T]。至少含 Φ_0。"""
    shaping = (phi_seq[-1] - phi_seq[0]) if len(phi_seq) >= 1 else 0.0
    coef = shaping_coef() if shaping_enabled() else 0.0
    return {
        "outcome_score": outcome,
        "shaping": shaping,
        "shaping_coef": coef,
        "final_score": outcome + coef * shaping,
    }


def resolve_phi(phi_by_turn: list[Optional[float]]) -> list[float]:
    """缺测轮进位(carry-forward):Φ_t 缺测 → 沿用 Φ_{t-1},该轮 ΔΦ=0,望远镜和不破。

    phi_by_turn[0] = Φ_0(初始 evaluate);缺测则兜底 0.0(与 phi_of 的 net_n=0 兜底同)。
    为何进位而非丢弃:丢弃会让轮次与 token 边界错位——0.10b 的 `phi_seq` 正是漏了这一步,
    轨迹级只取首尾故无害,步级则会把 r_t 记到错误的轮上(且 `phi_seq[-1]=final_phi`
    在末轮 eval 失败时覆盖的是更早某轮)。
    """
    out: list[float] = []
    last = phi_by_turn[0] if phi_by_turn and phi_by_turn[0] is not None else 0.0
    for v in phi_by_turn:
        if v is not None:
            last = v
        out.append(last)
    return out


def compute_step_rewards(outcome: float, phi_by_turn: list[Optional[float]], coef: float) -> list[float]:
    """步级 reward。返回长度 T 的列表，r[t-1] 对应 turn t(t=1..T)。

    phi_by_turn = [Φ_0, Φ_1, ..., Φ_T],**下标即轮次**,缺测填 None。T = len-1。
      r_t = coef × (Φ_t − Φ_{t-1})              t = 1..T-1
      ground: r_T = outcome + coef × (0 − Φ_{T-1})
      keep:   r_T = outcome + coef × (Φ_T − Φ_{T-1})
    T=0(无轮次)→ 返回 []，由调用方回退标量。
    """
    T = len(phi_by_turn) - 1
    if T <= 0:
        return []
    phi = resolve_phi(phi_by_turn)
    rewards = [coef * (phi[t] - phi[t - 1]) for t in range(1, T)]
    terminal_phi = phi[T] if step_terminal_mode() == "keep" else 0.0
    rewards.append(outcome + coef * (terminal_phi - phi[T - 1]))
    return rewards


def compute_step_episode_score(
    outcome: float, phi_by_turn: list[Optional[float]], coef: float
) -> float:
    """返回步级构造的轨迹总账，供正常散射与退化回退共同使用。

    这个接口刻意不读取 ``APPWORLD_SHAPING``。0.11d 的统一管道关闭轨迹级
    shaping，但步级构造仍由 ``APPWORLD_SHAPING_COEF`` 和终态口径控制；若回退
    到 ``compute_reward``，keep 臂会静默丢掉过程加分。

    无动作轮时没有可定义的末轮散射，按 outcome 回退；有动作轮时与
    ``sum(compute_step_rewards(...))`` 严格同口径。
    """
    rewards = compute_step_rewards(outcome, phi_by_turn, coef)
    return float(sum(rewards)) if rewards else float(outcome)

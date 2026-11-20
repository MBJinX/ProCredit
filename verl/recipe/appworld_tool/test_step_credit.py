"""Formula tests for per-turn rewards, terminal modes, and telescoping sums.
"""

import random

import pytest

from appworld_tool import aw_reward


COEF = 0.5


def telescoping_target(outcome: float, phi0: float, coef: float = COEF) -> float:
    """预注册不变量:Σ_t r_t = outcome − coef × Φ_0。"""
    return outcome - coef * phi0


# ---------------------------------------------------------------- 望远镜和

@pytest.mark.parametrize("outcome", [0.0, 1.0])
@pytest.mark.parametrize(
    "phi_by_turn",
    [
        [0.0, 0.5, 1.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.0, 0.0],
        [1.0, 0.5, 0.0, 0.75],
        [0.3],  # T=0 边界:无轮次
    ],
)
def test_telescoping_sum(outcome, phi_by_turn):
    r = aw_reward.compute_step_rewards(outcome, phi_by_turn, COEF)
    if len(phi_by_turn) == 1:
        assert r == [], "T=0 必须返回空,由调用方回退标量注入"
        return
    assert len(r) == len(phi_by_turn) - 1, "每轮恰好一个 r_t"
    assert sum(r) == pytest.approx(telescoping_target(outcome, phi_by_turn[0]))


def test_telescoping_sum_random_with_missing():
    """随机 Φ 序列 + 随机缺测,望远镜和仍严格成立(缺测进位不破不变量)。"""
    rng = random.Random(0)
    for _ in range(200):
        T = rng.randint(1, 12)
        phi = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(T + 1)]
        phi0 = phi[0]
        # 随机把非首位的若干轮打成缺测
        holed = [phi[0]] + [None if rng.random() < 0.4 else phi[t] for t in range(1, T + 1)]
        outcome = rng.choice([0.0, 1.0])
        r = aw_reward.compute_step_rewards(outcome, holed, COEF)
        assert sum(r) == pytest.approx(telescoping_target(outcome, phi0)), (holed, r)


def test_phi0_missing_defaults_zero():
    """Φ_0 缺测 → 兜底 0.0,望远镜和退化为 outcome。"""
    r = aw_reward.compute_step_rewards(1.0, [None, 0.5, 1.0], COEF)
    assert sum(r) == pytest.approx(1.0)


# ---------------------------------------------------------------- 终态归零

def test_terminal_zeroing_last_turn():
    """末轮 r_T = outcome + coef×(0 − Φ_{T-1});终局实测 Φ_T 不进 reward。"""
    phi = [0.0, 0.4, 0.9]
    r = aw_reward.compute_step_rewards(1.0, phi, COEF)
    assert r[-1] == pytest.approx(1.0 + COEF * (0.0 - phi[-2]))


def test_final_phi_does_not_enter_reward():
    """改动 Φ_T 不改变任何 r_t——严格保序的代价,与 0.10b 轨迹级信号来源不同。"""
    base = aw_reward.compute_step_rewards(1.0, [0.0, 0.4, 0.9], COEF)
    bumped = aw_reward.compute_step_rewards(1.0, [0.0, 0.4, 0.1], COEF)
    assert base == pytest.approx(bumped)


def test_intermediate_turns_are_pure_delta():
    phi = [0.0, 0.25, 0.75, 1.0]
    r = aw_reward.compute_step_rewards(0.0, phi, COEF)
    assert r[0] == pytest.approx(COEF * 0.25)
    assert r[1] == pytest.approx(COEF * 0.50)


# ---------------------------------------------------------------- 缺测进位

def test_carry_forward_zeroes_missing_turn():
    """缺测轮 ΔΦ=0,下一轮跨两轮补回。"""
    r = aw_reward.compute_step_rewards(0.0, [0.0, None, 0.6, 0.6], COEF)
    assert r[0] == pytest.approx(0.0), "缺测轮不给分"
    assert r[1] == pytest.approx(COEF * 0.6), "下一轮跨两轮补回"


def test_resolve_phi_carry_forward():
    assert aw_reward.resolve_phi([0.2, None, None, 0.8]) == [0.2, 0.2, 0.2, 0.8]
    assert aw_reward.resolve_phi([None, None]) == [0.0, 0.0]


# ---------------------------------------------------------------- 锚点

def test_anchor_order_and_dup_invariant():
    a = aw_reward.anchor_of(["req_b", "req_a"])
    b = aw_reward.anchor_of(["req_a", "req_b", "req_a"])
    assert a == b, "frozenset 语义:顺序无关 + 去重"


def test_anchor_distinguishes_progress():
    assert aw_reward.anchor_of(["r1"]) != aw_reward.anchor_of(["r1", "r2"])
    assert aw_reward.anchor_of([]) == aw_reward.anchor_of([]), "空集是合法且稳定的起手锚点"


def test_anchor_not_confused_by_substring():
    assert aw_reward.anchor_of(["ab", "c"]) != aw_reward.anchor_of(["a", "bc"]), "分隔符必须防拼接歧义"


# ---------------------------------------------------------------- 断言 id 抽取

EVAL_FIXTURE = {
    "passes": [
        {"requirement": "net_pass_1", "label": "no_op_fail"},
        {"requirement": "secondary_1", "label": "no_op_pass"},
    ],
    "failures": [
        {"requirement": "net_fail_1", "trace": "...", "label": "no_op_fail"},
        {"requirement": "noise_1", "trace": "...", "label": None},
    ],
    "success": False,
}


def test_net_requirements_filters_by_label():
    passed, all_net = aw_reward.net_requirements(EVAL_FIXTURE)
    assert passed == ["net_pass_1"]
    assert sorted(all_net) == ["net_fail_1", "net_pass_1"]


def test_net_requirements_agrees_with_net_counts():
    """与 0.10b 既有 net_counts 口径必须逐条一致(Φ 定义不能漂移)。"""
    passed, all_net = aw_reward.net_requirements(EVAL_FIXTURE)
    net_pass, net_n = aw_reward.net_counts(EVAL_FIXTURE)
    assert (len(passed), len(all_net)) == (net_pass, net_n)
    assert aw_reward.phi_of(EVAL_FIXTURE) == pytest.approx(net_pass / net_n)


# ---------------------------------------------------------------- 回归:0.10b 轨迹级口径不动

def test_legacy_trajectory_reward_unchanged(monkeypatch):
    monkeypatch.setenv("APPWORLD_SHAPING", "1")
    monkeypatch.setenv("APPWORLD_SHAPING_COEF", "0.5")
    s = aw_reward.compute_reward(1.0, [0.2, 0.5, 0.9])
    assert s["shaping"] == pytest.approx(0.7)
    assert s["final_score"] == pytest.approx(1.0 + 0.5 * 0.7)


def test_step_credit_off_by_default(monkeypatch):
    monkeypatch.delenv("APPWORLD_STEP_CREDIT", raising=False)
    assert aw_reward.step_credit_enabled() is False


# ---------------------------------------------------------------- 注入层(位置 + 折叠)

def _loop(response_length: int):
    """绕开 AgentLoopBase.__init__(要 verl rollout_config);_build_step_injection 只用这两个属性。"""
    aw_agent_loop = pytest.importorskip(
        "appworld_tool.aw_agent_loop", reason="需 verl 环境(from verl.experimental... import AgentLoopBase)"
    )
    obj = object.__new__(aw_agent_loop.AppWorldReactAgentLoop)
    obj.response_length = response_length
    obj.eval_every_turn = True
    return obj


@pytest.fixture
def step_on(monkeypatch):
    monkeypatch.setenv("APPWORLD_STEP_CREDIT", "1")
    monkeypatch.setenv("APPWORLD_SHAPING_COEF", "0.5")


def test_injection_positions_and_sum(step_on):
    lp = _loop(response_length=1000)
    phi = [0.0, 0.5, 1.0]          # T=2
    turn_end_pos = [100, 400]      # 两轮末 token
    pos, val = lp._build_step_injection(1.0, phi, turn_end_pos, resp_total=500, task_id="t")
    assert pos == [100, 400], "reward 必须落在每轮末 model token"
    assert sum(val) == pytest.approx(telescoping_target(1.0, phi[0]))
    assert val[-1] == pytest.approx(1.0 + COEF * (0.0 - phi[-2])), "末轮 = outcome + 终态归零项"


def test_injection_folds_truncated_turns_preserving_sum(step_on):
    """轮末 token 越过 response_length → 折叠进最后一个有效 token,望远镜和不破。"""
    lp = _loop(response_length=300)
    phi = [0.25, 0.5, 0.75, 1.0]   # T=3
    turn_end_pos = [100, 290, 900]  # 第三轮已被截断
    pos, val = lp._build_step_injection(1.0, phi, turn_end_pos, resp_total=1000, task_id="t")
    assert max(pos) <= 299, "不得散射到 response_length 之外"
    assert sum(val) == pytest.approx(telescoping_target(1.0, phi[0])), "折叠必须保住望远镜和"


def test_injection_folds_duplicate_positions(step_on):
    """空生成致两轮共用同一末 token 下标 → 求和折叠,不丢 reward。"""
    lp = _loop(response_length=1000)
    pos, val = lp._build_step_injection(0.0, [0.0, 0.5, 1.0], [7, 7], resp_total=50, task_id="t")
    assert pos == [7]
    assert sum(val) == pytest.approx(telescoping_target(0.0, 0.0))


def test_injection_degrades_on_misaligned_lengths(step_on):
    """turn_end_pos 与轮数不一致 = 对齐已破 → 回退标量,绝不静默错位注入。"""
    lp = _loop(response_length=1000)
    assert lp._build_step_injection(1.0, [0.0, 0.5, 1.0], [10], resp_total=50, task_id="t") == ([], [])


def test_injection_off_when_flag_unset(monkeypatch):
    monkeypatch.delenv("APPWORLD_STEP_CREDIT", raising=False)
    lp = _loop(response_length=1000)
    assert lp._build_step_injection(1.0, [0.0, 1.0], [10], resp_total=50, task_id="t") == ([], [])


def test_injection_requires_eval_every_turn(step_on):
    lp = _loop(response_length=1000)
    lp.eval_every_turn = False
    assert lp._build_step_injection(1.0, [0.0, 1.0], [10], resp_total=50, task_id="t") == ([], [])

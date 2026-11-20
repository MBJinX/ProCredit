"""Tests for trajectory and turn-level advantage estimation.

From the repository root: PYTHONPATH=verl:verl/recipe python -m pytest
verl/recipe/appworld_tool/test_gigpo_estimator.py
"""

import random
import statistics
import types

import numpy as np
import pytest
import torch

from appworld_tool import aw_reward
from appworld_tool.aw_gigpo import LAST_STATS, compute_appworld_phi_gigpo_advantage


CFG_NO_STD = types.SimpleNamespace(norm_adv_by_std_in_grpo=False)
CFG_STD = types.SimpleNamespace(norm_adv_by_std_in_grpo=True)


def obj_arr(items):
    a = np.empty(len(items), dtype=object)
    a[:] = items
    return a


def make_batch(rows, resp_len):
    """rows: list of dict(pos, val, anchors, tep, uid, masked=False)。
    返回 (token_level_rewards, response_mask, index, non_tensor_batch)。"""
    bsz = len(rows)
    tlr = torch.zeros(bsz, resp_len)
    mask = torch.ones(bsz, resp_len, dtype=torch.long)
    for i, r in enumerate(rows):
        for p, v in zip(r.get("pos", []), r.get("val", [])):
            tlr[i, p] = v
        if r.get("masked"):
            mask[i] = 0
        if r.get("scalar") is not None:  # 无步级表的标量注入(GRPO 回归锚用)
            tlr[i, resp_len - 1] = r["scalar"]
    ntb = {
        "step_reward_pos": obj_arr([r.get("pos", []) for r in rows]),
        "step_reward_val": obj_arr([r.get("val", []) for r in rows]),
        "anchor_by_turn": obj_arr([r.get("anchors") for r in rows]),
        "turn_end_pos": obj_arr([r.get("tep", []) for r in rows]),
        "loss_masked": obj_arr([bool(r.get("masked")) for r in rows]),
    }
    index = np.array([r["uid"] for r in rows], dtype=object)
    return tlr, mask, index, ntb


# ---------------------------------------------------------------- 锚点组内归一(手算对拍)

def test_anchor_group_norm_manual(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "0.5")
    g = 0.5
    # 两行同 uid,逐轮锚点相同(a0,a1);T=2
    rows = [
        dict(uid="u", pos=[1, 3], val=[0.2, 0.5], anchors=["a0", "a1", "aT"], tep=[1, 3]),
        dict(uid="u", pos=[2, 5], val=[0.4, 0.1], anchors=["a0", "a1", "aT"], tep=[2, 5]),
    ]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=6)
    adv, ret = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    assert torch.equal(adv, ret)
    assert LAST_STATS["fallback_turn_rtg"] is False
    # RTG
    A2, B2 = 0.5, 0.1
    A1, B1 = 0.2 + g * A2, 0.4 + g * B2
    # 组均值
    m0, m1 = (A1 + B1) / 2, (A2 + B2) / 2
    # episode 层(mean_norm,无 std):score=sum(val)
    sA, sB = 0.7, 0.5
    epA, epB = sA - 0.6, sB - 0.6
    # 行 A:记录1 覆盖 token0..1,记录2 覆盖 2..3,4..5 无 step 优势
    exp_A = torch.tensor([epA + (A1 - m0)] * 2 + [epA + (A2 - m1)] * 2 + [epA] * 2)
    exp_B = torch.tensor([epB + (B1 - m0)] * 3 + [epB + (B2 - m1)] * 3)
    assert torch.allclose(adv[0], exp_A, atol=1e-6), f"{adv[0]} vs {exp_A}"
    assert torch.allclose(adv[1], exp_B, atol=1e-6), f"{adv[1]} vs {exp_B}"


def test_different_anchors_no_pollution(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "0.5")
    # 3 行共享锚点 + 1 行独有锚点(组 size [3,3,1,1],中位=2 → 不触发 fallback)
    shared = dict(anchors=["a0", "a1", "aT"], tep=[1, 3])
    rows = [
        dict(uid="u", pos=[1, 3], val=[0.1, 0.2], **shared),
        dict(uid="u", pos=[1, 3], val=[0.3, 0.4], **shared),
        dict(uid="u", pos=[1, 3], val=[0.5, 0.6], **shared),
        dict(uid="u", pos=[1, 3], val=[9.0, 9.0], anchors=["b0", "b1", "bT"], tep=[1, 3]),
    ]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=5)
    adv, _ = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    assert LAST_STATS["fallback_turn_rtg"] is False
    # 独有锚点行 = 两个孤组 → step 优势为 0,只剩 episode 层
    s = [0.3, 0.7, 1.1, 18.0]
    ep3 = s[3] - statistics.mean(s)
    assert torch.allclose(adv[3], torch.full((5,), ep3), atol=1e-6)
    # 共享组的 step 均值不含 9.0(不被独有行污染):行 0 记录 2 的优势 = 0.2 - mean(0.2,0.4,0.6)
    ep0 = s[0] - statistics.mean(s)
    assert adv[0][3].item() == pytest.approx(ep0 + (0.2 - 0.4), abs=1e-6)


def test_fallback_median_and_pool(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    # 同 uid 两行,锚点互不相同 → 真锚点组全是孤组,中位 1 < 2 → 退化 turn-RTG 全池归一
    rows = [
        dict(uid="u", pos=[0, 2], val=[1.0, 1.0], anchors=["x0", "x1", "xT"], tep=[0, 2]),
        dict(uid="u", pos=[1, 3], val=[3.0, 1.0], anchors=["y0", "y1", "yT"], tep=[1, 3]),
    ]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=4)
    adv, _ = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    assert LAST_STATS["fallback_turn_rtg"] is True
    # 池 = 4 条记录 RTG:行0 [2,1] 行1 [4,1],均值 2
    # 行 0:记录1 覆盖 token0,记录2 覆盖 1..2,记录后 3 无 step
    s0, s1 = 2.0, 4.0
    ep0, ep1 = s0 - 3.0, s1 - 3.0
    assert adv[0][0].item() == pytest.approx(ep0 + (2 - 2), abs=1e-6)
    assert adv[0][1].item() == pytest.approx(ep0 + (1 - 2), abs=1e-6)
    assert adv[1][1].item() == pytest.approx(ep1 + (4 - 2), abs=1e-6)  # 记录1 覆盖 0..1
    assert adv[1][2].item() == pytest.approx(ep1 + (1 - 2), abs=1e-6)  # 记录2 覆盖 2..3
    assert adv[1][3].item() == pytest.approx(ep1 + (1 - 2), abs=1e-6)


# ---------------------------------------------------------------- loss_masked 排除

def test_loss_masked_excluded(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    shared = dict(anchors=["a0", "aT"], tep=[2])
    rows = [
        dict(uid="u", pos=[2], val=[1.0], **shared),
        dict(uid="u", pos=[2], val=[3.0], **shared),
        dict(uid="u", pos=[2], val=[100.0], masked=True, **shared),  # 被 mask 的天价分
    ]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=4)
    adv, _ = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    # mask 行优势全 0
    assert torch.equal(adv[2], torch.zeros(4))
    # episode 均值只含 1 和 3(=2),不被 100 污染;step 组均值同理(=2)
    assert adv[0][0].item() == pytest.approx((1 - 2) + (1 - 2), abs=1e-6)
    assert adv[1][0].item() == pytest.approx((3 - 2) + (3 - 2), abs=1e-6)
    assert LAST_STATS["n_masked_rows"] == 1
    assert LAST_STATS["n_step_rows"] == 2


# ---------------------------------------------------------------- RTG 折扣与广播边界

def test_rtg_discount_manual(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "0.5")
    rows = [dict(uid="u", pos=[0, 1, 2], val=[1.0, 1.0, 1.0], anchors=["a", "b", "c", "d"], tep=[0, 1, 2])]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=3)
    adv, _ = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    assert LAST_STATS["fallback_turn_rtg"] is True  # 单行孤组必退化
    rtg = [1 + 0.5 * (1 + 0.5 * 1), 1 + 0.5 * 1, 1.0]  # [1.75, 1.5, 1]
    m = statistics.mean(rtg)
    ep = 3.0 - 3.0  # 组内单行:episode 孤行约定 mean=自身 → 0
    for t in range(3):
        assert adv[0][t].item() == pytest.approx(ep + rtg[t] - m, abs=1e-6)


def test_truncation_folded_records(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    # 3 轮,后两轮末位越界折叠到 last_valid=5:散射表两条(2, 5);
    # 折叠记录的锚点 = 首个折叠轮动作前的 anchors[1]
    rows = [
        dict(uid="u", pos=[2, 5], val=[0.3, 0.7], anchors=["a0", "a1", "a2", "aT"], tep=[2, 9, 11]),
        dict(uid="u", pos=[2, 5], val=[0.5, 0.5], anchors=["a0", "a1", "a2", "aT"], tep=[2, 9, 11]),
    ]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=6)
    adv, _ = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    assert LAST_STATS["fallback_turn_rtg"] is False
    assert LAST_STATS["n_records"] == 4
    # 行 0:R2=0.7, R1=0.3+0.7=1.0;组均值 R1:(1.0,1.0)→1.0? 行1 R1=0.5+0.5=1.0,R2=0.5
    ep0 = 0.0  # 两行 score 均为 1.0 → episode 优势 0
    m2 = (0.7 + 0.5) / 2
    exp_r0 = [ep0 + (1.0 - 1.0)] * 3 + [ep0 + (0.7 - m2)] * 3
    assert torch.allclose(adv[0], torch.tensor(exp_r0), atol=1e-6)


def test_no_step_table_row_gets_episode_only():
    rows = [
        dict(uid="u", scalar=1.0),
        dict(uid="u", scalar=0.0),
    ]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=4)
    adv, _ = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    assert LAST_STATS["n_step_rows"] == 0
    assert torch.allclose(adv[0], torch.full((4,), 0.5), atol=1e-6)
    assert torch.allclose(adv[1], torch.full((4,), -0.5), atol=1e-6)


# ---------------------------------------------------------------- 与原生 GRPO 的回归锚

def test_episode_matches_native_grpo():
    core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")
    rng = random.Random(1)
    bsz, L = 8, 5
    rows = [dict(uid=f"u{i // 4}", scalar=rng.random()) for i in range(bsz)]
    tlr, mask, idx, ntb = make_batch(rows, L)
    ours, _ = compute_appworld_phi_gigpo_advantage(tlr.clone(), mask, index=idx, config=CFG_STD, non_tensor_batch=ntb)
    native, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=tlr.clone(), response_mask=mask.float(), index=idx, norm_adv_by_std_in_grpo=True
    )
    assert torch.allclose(ours, native, atol=1e-5)


def test_registered_and_signature():
    core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")
    import inspect

    import appworld_tool.aw_gigpo  # noqa: F401  触发注册

    fn = core_algos.get_adv_estimator_fn("appworld_phi_gigpo")
    assert fn is compute_appworld_phi_gigpo_advantage
    assert "non_tensor_batch" in inspect.signature(fn).parameters
    # gdpo 经反射分发仍拿得到两个参数(替换硬编码分支的兼容性锚)
    gdpo = core_algos.get_adv_estimator_fn("gdpo")
    p = inspect.signature(gdpo).parameters
    assert "non_tensor_batch" in p and "batch" in p


# ---------------------------------------------------------------- Φ_0 平移不变性(随机化)

def test_phi0_shift_invariance_random(monkeypatch):
    """episode 层对组内常数 Φ_0 平移不变(w=0 隔离 episode 层)。"""
    monkeypatch.setenv("APPWORLD_GIGPO_W", "0")
    rng = random.Random(7)
    for _ in range(100):
        n, T, L = rng.randint(2, 6), rng.randint(1, 6), 40
        coef = 0.5
        phi0 = rng.choice([0.0, 0.25, 0.5])
        delta = rng.choice([0.1, 0.25, -0.1])

        def build(shift):
            rows = []
            for j in range(n):
                phis = [min(1.0, max(0.0, phi0 + shift))] + [
                    rng2.choice([None, 0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(T)
                ]
                outcome = float(rng2.random() < 0.5)
                r = aw_reward.compute_step_rewards(outcome, phis, coef)
                tep = sorted(rng2.sample(range(L), len(r))) if r else []
                anchors = [f"g{j}_{t}" for t in range(T + 1)]
                rows.append(dict(uid="u", pos=tep, val=r, anchors=anchors, tep=tep))
            return make_batch(rows, L)

        rng2 = random.Random(rng.randint(0, 10**6))
        state = rng2.getstate()
        tlr1, mask1, idx1, ntb1 = build(0.0)
        rng2.setstate(state)  # 两次构造用同一随机流,只有 Φ_0 不同
        tlr2, mask2, idx2, ntb2 = build(delta)
        # mean-only 口径:平移不变性数学上对 std 归一同样成立,但组内分数全等时
        # (s−m)/(std+eps) 会把浮点尘埃放大 1e6 倍(原生 GRPO 同性质),数值上不可对拍
        a1, _ = compute_appworld_phi_gigpo_advantage(tlr1, mask1, index=idx1, config=CFG_NO_STD, non_tensor_batch=ntb1)
        a2, _ = compute_appworld_phi_gigpo_advantage(tlr2, mask2, index=idx2, config=CFG_NO_STD, non_tensor_batch=ntb2)
        assert torch.allclose(a1, a2, atol=1e-5), "episode 层必须对组内常数 Φ_0 平移不变"


# ---------------------------------------------------------------- 机制指标

def test_stats_fields(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    shared = dict(anchors=["a0", "aT"], tep=[1])
    rows = [dict(uid="u", pos=[1], val=[v], **shared) for v in (0.2, 0.8)]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=3)
    compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    for k in (
        "fallback_turn_rtg",
        "nonzero_var_group_frac",
        "anchor_group_size_median",
        "n_records",
        "n_none_anchor_records",
        "n_val_mismatch_rows",
    ):
        assert k in LAST_STATS
    assert LAST_STATS["nonzero_var_group_frac"] == 1.0  # 唯一真锚点组方差非零
    assert LAST_STATS["anchor_group_size_median"] == 2


def test_none_anchor_is_singleton(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    # 锚点缺测(None)不得与他人分组:两行 None 锚点 + 两行真锚点(维持中位≥2)
    rows = [
        dict(uid="u", pos=[1], val=[0.2], anchors=["a0", "aT"], tep=[1]),
        dict(uid="u", pos=[1], val=[0.8], anchors=["a0", "aT"], tep=[1]),
        dict(uid="u", pos=[1], val=[5.0], anchors=[None, "aT"], tep=[1]),
        dict(uid="u", pos=[1], val=[7.0], anchors=[None, "aT"], tep=[1]),
    ]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=3)
    adv, _ = compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    assert LAST_STATS["fallback_turn_rtg"] is False
    assert LAST_STATS["n_none_anchor_records"] == 2
    s = [0.2, 0.8, 5.0, 7.0]
    m = statistics.mean(s)
    # None 锚点行 step 优势为 0(孤组),只剩 episode
    assert torch.allclose(adv[2], torch.full((3,), 5.0 - m), atol=1e-5)
    assert torch.allclose(adv[3], torch.full((3,), 7.0 - m), atol=1e-5)
    # 真锚点两行互相成组
    assert adv[0][0].item() == pytest.approx((0.2 - m) + (0.2 - 0.5), abs=1e-5)


# ------------------------------------------------ 0.18a 新增：C4 全胜组占比 + 长度偏置仪表
# 这两条是纯观测指标，但**必须有测**：0.14a 把 27B 的「摊步 − 终点 = −8.04」归因到优势层
# 长度偏置，靠的就是事后重算全胜组占比与长度-优势相关；0.18a 把它们做进估计器实时落盘，
# 若指标算错，摊步臂的报警就是假的。这里调真估计器、手算对拍三种组结构。

def _stats_for(outcomes, monkeypatch, resp_lens=None):
    """构造 len(outcomes) 行、同一 uid 的一组，按 outcome_score 打标，返回 LAST_STATS。"""
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    shared = dict(anchors=["a0", "aT"], tep=[1])
    rows = [dict(uid="g", pos=[1], val=[0.1 * (i + 1)], **shared) for i in range(len(outcomes))]
    tlr, mask, idx, ntb = make_batch(rows, resp_len=6)
    if resp_lens is not None:  # 让各行有效长度不同，才能测长度-优势相关
        for i, L in enumerate(resp_lens):
            mask[i, L:] = 0
    ntb["outcome_score"] = np.array([float(o) for o in outcomes], dtype=np.float64)
    compute_appworld_phi_gigpo_advantage(tlr, mask, index=idx, config=CFG_NO_STD, non_tensor_batch=ntb)
    return dict(LAST_STATS)


def test_c4_all_win_group_frac(monkeypatch):
    s = _stats_for([1.0, 1.0, 1.0], monkeypatch)
    assert s["n_valid_groups"] == 1
    assert s["n_all_win_groups"] == 1
    assert s["all_win_group_frac"] == 1.0
    assert s["n_all_failure_groups"] == 0
    assert s["all_failure_group_frac"] == 0.0


def test_c4_all_failure_group_unchanged(monkeypatch):
    # 死组计数是老指标，改动不能动它（这条就是防回归的）
    s = _stats_for([0.0, 0.0, 0.0], monkeypatch)
    assert s["n_all_win_groups"] == 0
    assert s["all_win_group_frac"] == 0.0
    assert s["n_all_failure_groups"] == 1
    assert s["all_failure_group_frac"] == 1.0


def test_c4_mixed_group_counts_neither(monkeypatch):
    s = _stats_for([1.0, 0.0], monkeypatch)
    assert s["n_valid_groups"] == 1
    assert s["n_all_win_groups"] == 0
    assert s["n_all_failure_groups"] == 0


def test_len_adv_pearson_sign(monkeypatch):
    # 三行、Φ 分递增(0.1/0.2/0.3)→ episode 优势也递增；把长度设成递减，
    # 相关必须是负的（这正是 0.14a 里「短=优」那条相关的符号）。
    s = _stats_for([0.0, 0.0, 0.0], monkeypatch, resp_lens=[6, 4, 2])
    assert s["len_episode_adv_pearson"] < -0.9
    s2 = _stats_for([0.0, 0.0, 0.0], monkeypatch, resp_lens=[2, 4, 6])
    assert s2["len_episode_adv_pearson"] > 0.9

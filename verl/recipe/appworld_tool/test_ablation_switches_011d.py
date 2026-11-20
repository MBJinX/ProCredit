"""0.11d 四臂 reward 构造消融的行为与真通路测试。"""

import types

import numpy as np
import pytest
import torch

from appworld_tool import aw_reward
from appworld_tool.aw_gigpo import LAST_STATS, compute_appworld_phi_gigpo_advantage


def _obj_arr(rows):
    arr = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        arr[i] = row
    return arr


def _step_batch(rows, response_mask=None):
    """rows: uid/pos/val/anchors/turn_end_pos。"""
    resp_len = max(max(row["pos"]) for row in rows) + 1
    rewards = torch.zeros(len(rows), resp_len)
    for i, row in enumerate(rows):
        for pos, value in zip(row["pos"], row["val"]):
            rewards[i, pos] = value
    mask = response_mask if response_mask is not None else torch.ones_like(rewards)
    ntb = {
        "step_reward_pos": _obj_arr([row["pos"] for row in rows]),
        "step_reward_val": _obj_arr([row["val"] for row in rows]),
        "anchor_by_turn": _obj_arr([row["anchors"] for row in rows]),
        "turn_end_pos": _obj_arr([row["turn_end_pos"] for row in rows]),
    }
    uid = np.array([row["uid"] for row in rows], dtype=object)
    return rewards, mask, uid, ntb


def test_keep_terminal_preserves_final_progress(monkeypatch):
    """keep 终态不扣回，逐轮和必须包含最终进度增量。"""
    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", "keep")
    outcome = 0.0
    coef = 0.5
    phi = [0.0, 0.25, 0.75]

    rewards = aw_reward.compute_step_rewards(outcome, phi, coef)

    assert rewards[-1] == pytest.approx(outcome + coef * (phi[-1] - phi[-2]))
    assert sum(rewards) == pytest.approx(outcome + coef * (phi[-1] - phi[0]))


def test_keep_terminal_flows_through_agent_loop_injection(monkeypatch):
    """训练真正调用的 _build_step_injection 必须消费 keep 开关。"""
    aw_agent_loop = pytest.importorskip("appworld_tool.aw_agent_loop", reason="需要 verl")
    monkeypatch.setenv("APPWORLD_STEP_CREDIT", "1")
    monkeypatch.setenv("APPWORLD_SHAPING_COEF", "0.5")
    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", "keep")
    loop = object.__new__(aw_agent_loop.AppWorldReactAgentLoop)
    loop.response_length = 32
    loop.eval_every_turn = True

    pos, values = loop._build_step_injection(
        0.0, [0.0, 0.25, 0.75], [4, 9], resp_total=12, task_id="task"
    )

    assert pos == [4, 9]
    assert sum(values) == pytest.approx(0.5 * 0.75)


def test_terminal_bonus_episode_matches_trajectory_shaping(monkeypatch):
    """W=0 时，keep 散射的 episode 层与 0.11a 轨迹级公式逐条等值。"""
    pytest.importorskip("verl", reason="需要运行时 verl 的真散射通路")
    from appworld_tool.test_scatter_integration import _mk_output, _postprocess

    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", "keep")
    monkeypatch.setenv("APPWORLD_GIGPO_W", "0")
    coef = 0.5
    phi0 = 0.125  # Φ₀=0 是当前数据事实，不是公式公理；测试必须锁住减项。
    phi_t = [0.125, 0.25, 0.5, 1.0]
    outcomes = [0.0, 0.0, 0.0, 1.0]
    outputs = []
    expected_scores = []
    for outcome, final_phi in zip(outcomes, phi_t):
        phi = [phi0, (phi0 + final_phi) / 2, final_phi]
        rewards = aw_reward.compute_step_rewards(outcome, phi, coef)
        outputs.append(
            _mk_output(
                prompt_len=2,
                resp_len=8,
                n_valid=6,
                reward_score=sum(rewards),
                step_pos=[1, 5],
                step_val=rewards,
            )
        )
        expected_scores.append(outcome + coef * (final_phi - phi[0]))

    batch = _postprocess(outputs)
    tensor_scores = batch.batch["rm_scores"].sum(dim=-1, dtype=torch.float64)
    assert tensor_scores.tolist() == pytest.approx(expected_scores)

    cfg = types.SimpleNamespace(norm_adv_by_std_in_grpo=True)
    advantages, _ = compute_appworld_phi_gigpo_advantage(
        token_level_rewards=batch.batch["rm_scores"],
        response_mask=batch.batch["response_mask"],
        index=np.array(["same-task"] * len(outputs), dtype=object),
        config=cfg,
        non_tensor_batch=batch.non_tensor_batch,
    )
    expected = torch.tensor(expected_scores, dtype=torch.float64)
    expected = (expected - expected.mean()) / expected.std()
    observed = advantages[:, 0].double()
    assert observed.tolist() == pytest.approx(expected.tolist(), abs=1e-5)


def test_keep_full_production_chain_build_postprocess_estimator(monkeypatch):
    """keep 必须贯穿生产构造、运行时散射与真实估计器，防止只测拼好的张量。"""
    aw_agent_loop = pytest.importorskip("appworld_tool.aw_agent_loop", reason="需要 verl")
    from appworld_tool.test_scatter_integration import _mk_output, _postprocess

    monkeypatch.setenv("APPWORLD_STEP_CREDIT", "1")
    monkeypatch.setenv("APPWORLD_SHAPING_COEF", "0.5")
    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", "keep")
    monkeypatch.setenv("APPWORLD_GIGPO_GROUP", "task")
    monkeypatch.setenv("APPWORLD_GIGPO_W", "1")
    loop = object.__new__(aw_agent_loop.AppWorldReactAgentLoop)
    loop.response_length = 8
    loop.eval_every_turn = True

    specs = [
        (0.0, [0.2, 0.5, 0.8], ["a0", "a1", "a2"]),
        (1.0, [0.2, 0.2, 0.4], [None, "b1", "b2"]),
    ]
    outputs = []
    expected_scores = []
    for outcome, phi, anchors in specs:
        pos, val = loop._build_step_injection(outcome, phi, [2, 5], 6, "task")
        expected_scores.append(outcome + 0.5 * (phi[-1] - phi[0]))
        out = _mk_output(2, 8, 6, sum(val), pos, val)
        out.extra_fields.update(
            {
                "anchor_by_turn": anchors,
                "turn_end_pos": [2, 5],
                "loss_masked": False,
                "extra_info": {"task_id": "task"},
            }
        )
        outputs.append(out)

    batch = _postprocess(outputs)
    assert batch.batch["rm_scores"].sum(-1, dtype=torch.float64).tolist() == pytest.approx(
        expected_scores
    )
    advantages, _ = compute_appworld_phi_gigpo_advantage(
        batch.batch["rm_scores"],
        batch.batch["response_mask"],
        index=np.array(["task", "task"], dtype=object),
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=False),
        non_tensor_batch=batch.non_tensor_batch,
    )
    assert torch.isfinite(advantages).all()
    assert LAST_STATS["n_step_rows"] == 2
    assert LAST_STATS["group_mode"] == "task"


def test_keep_degraded_injection_falls_back_to_keep_score(monkeypatch):
    """轮界错位时没有散射表，但标量总账仍保留 keep 过程分。"""
    aw_agent_loop = pytest.importorskip("appworld_tool.aw_agent_loop", reason="需要 verl")
    monkeypatch.setenv("APPWORLD_STEP_CREDIT", "1")
    monkeypatch.setenv("APPWORLD_SHAPING_COEF", "0.5")
    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", "keep")
    monkeypatch.setenv("APPWORLD_SHAPING", "0")
    loop = object.__new__(aw_agent_loop.AppWorldReactAgentLoop)
    loop.response_length = 8
    loop.eval_every_turn = True

    pos, val = loop._build_step_injection(0.0, [0.2, 0.5, 0.8], [2], 6, "bad")
    fallback = aw_reward.compute_step_episode_score(0.0, [0.2, 0.5, 0.8], 0.5)

    assert (pos, val) == ([], [])
    assert fallback == pytest.approx(0.5 * (0.8 - 0.2))
    assert fallback != aw_reward.compute_reward(0.0, [0.2, 0.8])["final_score"]


@pytest.mark.parametrize("terminal_mode", ["ground", "keep"])
def test_zero_coef_scatter_is_outcome_only(monkeypatch, terminal_mode):
    """COEF=0 时终态开关无效，散射严格退化成 [0,...,0,y]。"""
    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", terminal_mode)
    outcome = 1.0
    rewards = aw_reward.compute_step_rewards(outcome, [0.0, 0.25, None, 1.0], coef=0.0)
    assert rewards == [0.0, 0.0, outcome]


def test_task_group_is_explicit_and_does_not_use_anchor_fallback(monkeypatch):
    """task 模式直接按 uid 分组，即使锚点全是孤组也不得伪装成 fallback。"""
    monkeypatch.setenv("APPWORLD_GIGPO_GROUP", "task")
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    rows = [
        {"uid": "u", "pos": [0, 2], "val": [1.0, 1.0], "anchors": ["a", "b"], "turn_end_pos": [0, 2]},
        {"uid": "u", "pos": [1, 2], "val": [3.0, 1.0], "anchors": ["x", "y"], "turn_end_pos": [1, 2]},
    ]
    rewards, mask, uid, ntb = _step_batch(rows)

    compute_appworld_phi_gigpo_advantage(
        rewards,
        mask,
        index=uid,
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=False),
        non_tensor_batch=ntb,
    )

    assert LAST_STATS["group_mode"] == "task"
    assert LAST_STATS["fallback_turn_rtg"] is False


def test_task_group_with_none_anchor_matches_manual_advantage(monkeypatch):
    """task 模式必须纳入 anchor=None 记录，且轮级广播与手算一致。"""
    monkeypatch.setenv("APPWORLD_GIGPO_GROUP", "task")
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "1.0")
    monkeypatch.setenv("APPWORLD_GIGPO_W", "1.0")
    rows = [
        {"uid": "u", "pos": [1, 3], "val": [2.0, 0.0], "anchors": [None, "a"], "turn_end_pos": [1, 3]},
        {"uid": "u", "pos": [1, 3], "val": [0.0, 2.0], "anchors": [None, "b"], "turn_end_pos": [1, 3]},
    ]
    rewards, mask, uid, ntb = _step_batch(rows)

    advantages, _ = compute_appworld_phi_gigpo_advantage(
        rewards,
        mask,
        index=uid,
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=False),
        non_tensor_batch=ntb,
    )

    expected = torch.tensor([[0.5, 0.5, -1.5, -1.5], [0.5, 0.5, 0.5, 0.5]])
    assert torch.allclose(advantages, expected)
    assert LAST_STATS["n_none_anchor_records"] == 2
    assert LAST_STATS["fallback_turn_rtg"] is False


def test_folded_turns_sharing_token_preserve_keep_sum(monkeypatch):
    """截断折叠与多轮同 token 同时发生时，总账不得丢失。"""
    aw_agent_loop = pytest.importorskip("appworld_tool.aw_agent_loop", reason="需要 verl")
    monkeypatch.setenv("APPWORLD_STEP_CREDIT", "1")
    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", "keep")
    monkeypatch.setenv("APPWORLD_SHAPING_COEF", "0.5")
    loop = object.__new__(aw_agent_loop.AppWorldReactAgentLoop)
    loop.response_length = 4
    loop.eval_every_turn = True

    pos, val = loop._build_step_injection(1.0, [0.1, 0.3, 0.6, 0.9], [1, 9, 10], 11, "fold")

    assert pos == [1, 3]
    assert len(val) == 2
    assert sum(val) == pytest.approx(1.0 + 0.5 * (0.9 - 0.1))


def test_neardup_and_infra_mask_interaction(monkeypatch):
    """近同分保护不能被 infra 行污染，infra 行优势必须全零。"""
    monkeypatch.setenv("APPWORLD_GIGPO_W", "0")
    rewards = torch.tensor([[0.0], [1e-7], [1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0], [1.0], [0.0]])
    ntb = {"loss_masked": _obj_arr([False, False, True])}

    advantages, _ = compute_appworld_phi_gigpo_advantage(
        rewards,
        mask,
        index=np.array(["u", "u", "u"], dtype=object),
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=True),
        non_tensor_batch=ntb,
    )

    assert torch.count_nonzero(advantages) == 0
    assert LAST_STATS["n_masked_rows"] == 1
    assert LAST_STATS["n_episode_neardup_groups"] == 1
    assert LAST_STATS["episode_neardup_raw_adv_absmax"] > 0


def test_unknown_step_terminal_raises(monkeypatch):
    monkeypatch.setenv("APPWORLD_STEP_TERMINAL", "mystery")
    with pytest.raises(ValueError, match="APPWORLD_STEP_TERMINAL"):
        aw_reward.compute_step_rewards(0.0, [0.0, 0.5], 0.5)


def test_unknown_gigpo_group_raises(monkeypatch):
    monkeypatch.setenv("APPWORLD_GIGPO_GROUP", "mystery")
    with pytest.raises(ValueError, match="APPWORLD_GIGPO_GROUP"):
        compute_appworld_phi_gigpo_advantage(
            torch.zeros(2, 1),
            torch.ones(2, 1),
            index=np.array(["u", "u"], dtype=object),
        )


def test_neardup_guard_records_raw_values_before_zeroing(monkeypatch):
    """近同分保护归零前的原始相对分、分数极差和受保护组数必须可审计。"""
    monkeypatch.setenv("APPWORLD_GIGPO_W", "0")
    rewards = torch.tensor([[0.0], [1e-7]], dtype=torch.float32)
    mask = torch.ones_like(rewards)

    compute_appworld_phi_gigpo_advantage(
        rewards,
        mask,
        index=np.array(["u", "u"], dtype=object),
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=True),
        non_tensor_batch={},
    )

    assert LAST_STATS["n_episode_neardup_groups"] == 1
    assert LAST_STATS["episode_neardup_score_range_max"] > 0
    assert LAST_STATS["episode_neardup_raw_adv_absmax"] > 0
    assert LAST_STATS["episode_neardup_leak_absmax"] == 0


def test_c2_all_failure_differentiated_group_is_observed(monkeypatch):
    """C2 必须从估计器真 score/adv 记录“可分组已复活”，不能事后猜。"""
    monkeypatch.setenv("APPWORLD_GIGPO_W", "0")
    rewards = torch.tensor([[0.0], [0.25]], dtype=torch.float32)
    mask = torch.ones_like(rewards)
    ntb = {"outcome_score": _obj_arr([0.0, 0.0])}

    compute_appworld_phi_gigpo_advantage(
        rewards,
        mask,
        index=np.array(["u", "u"], dtype=object),
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=False),
        non_tensor_batch=ntb,
    )

    assert LAST_STATS["n_all_failure_groups"] == 1
    assert LAST_STATS["n_all_failure_shaping_diff_groups"] == 1
    assert LAST_STATS["n_all_failure_diff_groups_episode_nonzero"] == 1
    assert LAST_STATS["n_all_failure_episode_nonzero_rows"] == 2


def test_unique_variance_is_computed_after_trajectory_aggregation(monkeypatch):
    """同轨迹驻留造成的轮间方差不能冒充不同轨迹之间的方差。"""
    monkeypatch.setenv("APPWORLD_GIGPO_GROUP", "anchor")
    monkeypatch.setenv("APPWORLD_GIGPO_GAMMA", "0")
    rows = [
        {"uid": "u", "pos": [0, 1], "val": [2.0, 0.0], "anchors": ["same", "same"], "turn_end_pos": [0, 1]},
        {"uid": "u", "pos": [0, 1], "val": [0.0, 2.0], "anchors": ["same", "same"], "turn_end_pos": [0, 1]},
    ]
    rewards, mask, uid, ntb = _step_batch(rows)

    compute_appworld_phi_gigpo_advantage(
        rewards,
        mask,
        index=uid,
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=False),
        non_tensor_batch=ntb,
    )

    assert LAST_STATS["nonzero_var_group_frac"] == 1.0, "轮记录本身有方差（前提自检）"
    assert LAST_STATS["nonzero_var_group_frac_unique"] == 0.0, (
        "两条轨迹各自 RTG 均值同为 1，按轨迹聚合后不应再报告非零方差"
    )


def test_update_model_tokens_are_dumped_per_turn(monkeypatch):
    """每轮实际进入更新的模型 token 数来自真实 response_mask 区间。"""
    monkeypatch.setenv("APPWORLD_GIGPO_GROUP", "task")
    rows = [
        {"uid": "u", "pos": [2, 5], "val": [0.2, 0.8], "anchors": ["a", "b"], "turn_end_pos": [2, 5]},
        {"uid": "u", "pos": [1, 5], "val": [0.4, 0.6], "anchors": ["x", "y"], "turn_end_pos": [1, 5]},
    ]
    response_mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    rewards, mask, uid, ntb = _step_batch(rows, response_mask=response_mask)
    ntb["extra_info"] = _obj_arr([{"task_id": "task-a"}, {"task_id": "task-a"}])

    compute_appworld_phi_gigpo_advantage(
        rewards,
        mask,
        index=uid,
        config=types.SimpleNamespace(norm_adv_by_std_in_grpo=False),
        non_tensor_batch=ntb,
    )

    token_rows = LAST_STATS["update_token_rows"]
    assert token_rows[0]["task_id"] == "task-a"
    assert token_rows[0]["tokens_by_turn"] == [2, 3]
    assert token_rows[1]["tokens_by_turn"] == [2, 3]
    assert LAST_STATS["update_model_tokens_total"] == 10

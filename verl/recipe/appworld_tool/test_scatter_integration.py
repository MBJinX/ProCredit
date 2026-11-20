"""Integration tests for the actual AgentLoopWorker._postprocess path.

Verify per-turn reward placement, scalar fallback, and conservation of the
injected reward sum. These tests call the framework implementation directly.
"""

import numpy as np
import pytest
import torch

verl = pytest.importorskip("verl", reason="需要 verl 环境")
from verl.experimental.agent_loop.agent_loop import (  # noqa: E402
    AgentLoopMetrics,
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)


def _mk_output(prompt_len, resp_len, n_valid, reward_score, step_pos=None, step_val=None):
    """造一条 _InternalAgentLoopOutput(右填充,前 n_valid 个 response token 有效)。

    字段按 agent_loop.py:148-170 的真实定义给全(input_ids=prompt+response 拼接)。
    每条 output 的张量都带 batch 维 `[1, len]`——_postprocess 用 `torch.cat(dim=0)`
    而非 stack(agent_loop.py:936-941),给 1D 张量会 cat 成 1D 并炸 TensorDict batch_size。
    """
    extra = {}
    if step_pos is not None:
        extra["step_reward_pos"] = step_pos
        extra["step_reward_val"] = step_val
    total = prompt_len + resp_len
    attn = torch.zeros(1, total, dtype=torch.long)
    attn[0, :prompt_len] = 1                       # prompt 全有效
    attn[0, prompt_len : prompt_len + n_valid] = 1  # response 前 n_valid 有效
    prompt_ids = torch.zeros(1, prompt_len, dtype=torch.long)
    response_ids = torch.zeros(1, resp_len, dtype=torch.long)
    resp_mask = torch.cat([torch.ones(n_valid), torch.zeros(resp_len - n_valid)]).long().unsqueeze(0)
    return _InternalAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=torch.cat([prompt_ids, response_ids], dim=1),
        response_mask=resp_mask,
        attention_mask=attn,
        position_ids=torch.arange(total, dtype=torch.long).unsqueeze(0),
        num_turns=len(step_pos) if step_pos else 1,
        reward_score=reward_score,
        metrics=AgentLoopMetrics(),
        extra_fields=extra,
    )


class _StubWorker:
    """AgentLoopWorker.__init__ 要 ray/config,_postprocess 只用到这一个属性(agent_loop.py:990)。"""

    reward_loop_worker_handles = None


def _postprocess(outputs):
    """直接调用未绑定的 _postprocess,喂最小 stub self。"""
    return AgentLoopWorker._postprocess(_StubWorker(), outputs)


def test_scatter_happens_in_runtime_verl():
    """① 逐轮散射生效:reward 精确落在轮末位置。"""
    pos, val = [3, 7, 11], [0.25, -0.5, 1.25]
    out = _mk_output(prompt_len=4, resp_len=16, n_valid=12, reward_score=sum(val), step_pos=pos, step_val=val)
    batch = _postprocess([out])
    rm = batch.batch["rm_scores"][0]

    for p, v in zip(pos, val):
        assert rm[p].item() == pytest.approx(v), (
            f"位置 {p} 期望 {v} 实得 {rm[p].item()}——散射块不在运行时那棵 verl 里?"
            f"(检查 verl/experimental/agent_loop/agent_loop.py 的 _postprocess)"
        )
    zeros = [i for i in range(rm.numel()) if i not in pos]
    assert torch.allclose(rm[zeros], torch.zeros(len(zeros))), "非轮末位置必须为 0"


def test_no_step_fields_falls_back_to_scalar():
    """② 无 step_reward_pos 的 recipe:退回标量放最后一个有效 token,零影响。"""
    out = _mk_output(prompt_len=4, resp_len=16, n_valid=12, reward_score=0.75)
    batch = _postprocess([out])
    rm = batch.batch["rm_scores"][0]
    assert rm[11].item() == pytest.approx(0.75), "标量应落在最后一个有效 token(下标 n_valid-1)"
    assert rm.sum().item() == pytest.approx(0.75), "别处必须为 0"


def test_telescoping_sum_preserved_after_scatter():
    """③ 散射后逐行求和 == 注入值之和(estimator 的 episode 层依赖这一点)。"""
    pos, val = [2, 5, 9], [0.1, 0.2, -0.05]
    out = _mk_output(prompt_len=3, resp_len=12, n_valid=10, reward_score=sum(val), step_pos=pos, step_val=val)
    batch = _postprocess([out])
    assert batch.batch["rm_scores"][0].sum().item() == pytest.approx(sum(val))


def test_mixed_batch_rows_independent():
    """混合 batch:有/无稀疏表的行互不干扰(主矩阵会同时有 infra 行)。"""
    pos, val = [1, 4], [0.3, 0.7]
    a = _mk_output(4, 12, 8, sum(val), pos, val)   # 步级行
    b = _mk_output(4, 12, 6, 0.9)                  # 标量行
    batch = _postprocess([a, b])
    rm = batch.batch["rm_scores"]
    assert rm[0, 1].item() == pytest.approx(0.3) and rm[0, 4].item() == pytest.approx(0.7)
    assert rm[0].sum().item() == pytest.approx(1.0)
    assert rm[1, 5].item() == pytest.approx(0.9), "标量行落在其自身最后有效 token(n_valid-1=5)"
    assert rm[1].sum().item() == pytest.approx(0.9)


def test_out_of_range_positions_ignored():
    """越界位置被丢弃,不炸也不污染(截断轮折叠的边界保护)。"""
    pos, val = [2, 999], [0.5, 0.5]
    out = _mk_output(4, 12, 8, 1.0, pos, val)
    batch = _postprocess([out])
    rm = batch.batch["rm_scores"][0]
    assert rm[2].item() == pytest.approx(0.5)
    assert rm.sum().item() == pytest.approx(0.5), "越界项被丢,和只剩合法项"

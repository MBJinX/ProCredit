# Copyright 2025 Nanyang Technological University (NTU), Singapore and the verl-agent (GiGPO) team.
# Copyright 2026 AppWorld phase0.11b adaptation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from verl-agent/gigpo/core_gigpo.py (GiGPO,
# https://arxiv.org/abs/2505.10978). 原实现的批结构是「每行 = 一个 env step」;
# 本适配是「每行 = 一条完整多轮轨迹」,step 层按 (row, turn) 记录展开。
"""Trajectory-level and turn-level advantage estimator for ProCredit.

Registered as appworld_phi_gigpo. The trajectory term normalizes summed
rewards within each task group. The turn term computes reward-to-go,
centers it within a configured group, and broadcasts it to the turn's
model tokens. Infrastructure-masked trajectories are excluded from group
statistics and receive zero advantage.

ProCredit uses gamma=1, group=task, mode=mean_norm, and weight=1 together
with the keep terminal reward mode. Historical defaults additionally
support anchor grouping and discounted returns; set the paper configuration
explicitly. Adaptive suppression, group-dependent discounting, and token
spreading are optional ablations, disabled for ProCredit.
"""

import json
import logging
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 最近一次调用的机制指标(smoke/D1' 判读用;driver 进程内单线程调用,无竞态)
LAST_STATS: dict[str, Any] = {}
# 对账键(0.11c 审查 P0-5):driver 进程内估计器调用序号。resume 后从 1 重计,
# 与 APPWORLD_RUN_ATTEMPT(run 脚本按重启递增导出)联合构成 (attempt_id, call_seq)
# 显式 join 键——判决脚本不再依赖「行序=步序」的隐式假设。
_CALL_SEQ = 0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _rows_records(
    token_level_rewards: torch.Tensor,
    non_tensor_batch: dict,
    valid: list[bool],
) -> tuple[list[Optional[list[dict]]], int, int]:
    """把每行的步级散射表展开成记录序列。

    返回 (per_row_records, n_step_rows, n_val_mismatch)。
    per_row_records[i] = None(该行无步级表/不合法)或
      [{pos, reward, anchor}...] 按 pos 升序。
    """
    bsz, resp_len = token_level_rewards.shape
    pos_arr = non_tensor_batch.get("step_reward_pos")
    val_arr = non_tensor_batch.get("step_reward_val")
    anchor_arr = non_tensor_batch.get("anchor_by_turn")
    tep_arr = non_tensor_batch.get("turn_end_pos")

    out: list[Optional[list[dict]]] = [None] * bsz
    n_step_rows = 0
    n_mismatch = 0
    if pos_arr is None or tep_arr is None:
        return out, 0, 0

    for i in range(bsz):
        if not valid[i]:
            continue
        pos = pos_arr[i]
        tep = tep_arr[i]
        anchors = anchor_arr[i] if anchor_arr is not None else None
        if not pos or not tep:
            continue
        # 散射折叠的逆映射:capped pos → 首个折叠进该 pos 的轮 t(1-indexed)
        first_turn_of_pos: dict[int, int] = {}
        last_valid = resp_len - 1
        for t, p in enumerate(tep, start=1):
            cp = min(int(p), last_valid)
            first_turn_of_pos.setdefault(cp, t)
        records = []
        broken = False
        for p in pos:
            p = int(p)
            t0 = first_turn_of_pos.get(p)
            if t0 is None or p < 0 or p >= resp_len:
                broken = True  # 与注入坐标系不一致:整行退出 step 层,绝不静默错位
                break
            # 记录锚点 = 首个折叠轮动作前的状态等价类 anchor_by_turn[t0-1]
            anchor = None
            if anchors is not None and 0 <= t0 - 1 < len(anchors):
                anchor = anchors[t0 - 1]
            records.append({"pos": p, "reward": float(token_level_rewards[i, p].item()), "anchor": anchor})
        if broken or not records:
            if broken:
                logger.warning("appworld_phi_gigpo: 行 %d 散射表与 turn_end_pos 不一致,该行退出 step 层", i)
            continue
        records.sort(key=lambda r: r["pos"])
        # 与注入值对拍(KL-in-reward 等改写 token reward 的配置会失配 → 告警但仍以张量为准)
        if val_arr is not None and val_arr[i]:
            inj = {int(p): float(v) for p, v in zip(pos, val_arr[i])}
            for r in records:
                if abs(inj.get(r["pos"], float("nan")) - r["reward"]) > 1e-4:
                    n_mismatch += 1
                    break
        out[i] = records
        n_step_rows += 1
    return out, n_step_rows, n_mismatch


def compute_appworld_phi_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
    non_tensor_batch: Optional[dict] = None,
    meta_info: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """见模块 docstring。返回 (advantages, returns),二者相同(GRPO/GiGPO 约定)。"""
    if index is None:
        raise ValueError("appworld_phi_gigpo 需要 uid 分组(non_tensor_batch['uid'] 缺失)")
    non_tensor_batch = non_tensor_batch or {}
    bsz, resp_len = token_level_rewards.shape

    norm_by_std = True
    if config is not None:
        norm_by_std = bool(getattr(config, "norm_adv_by_std_in_grpo", True))
    gamma = _env_float("APPWORLD_GIGPO_GAMMA", 0.95)
    w = _env_float("APPWORLD_GIGPO_W", 1.0)
    mode = os.environ.get("APPWORLD_GIGPO_MODE", "mean_norm")
    if mode not in ("mean_norm", "mean_std_norm"):
        raise ValueError(f"未知 APPWORLD_GIGPO_MODE: {mode}")
    group_mode = os.environ.get("APPWORLD_GIGPO_GROUP", "anchor")
    if group_mode not in ("anchor", "task"):
        raise ValueError(f"未知 APPWORLD_GIGPO_GROUP: {group_mode}")
    # 0.19a 自适应摊步开关。取值只认 "0"/"1":口径开关拼错(如 "true"/"on")在
    # `== "1"` 写法下会静默落回 0.18a 行为,那正是「自适应臂其实没自适应」的静默退化。
    adaptive_raw = os.environ.get("APPWORLD_ADAPTIVE_SPREAD", "0")
    if adaptive_raw not in ("0", "1"):
        raise ValueError(f"APPWORLD_ADAPTIVE_SPREAD 只能是 0|1,实得 {adaptive_raw!r}")
    adaptive_spread = adaptive_raw == "1"
    # 0.22a 按组换 γ 开关。取值纪律与 ADAPTIVE_SPREAD 相同：拼错静默落回默认
    # 就是「成本臂其实没路由」的静默退化，只认 "0"/"1"。
    route_raw = os.environ.get("APPWORLD_GROUP_GAMMA_ROUTE", "0")
    if route_raw not in ("0", "1"):
        raise ValueError(f"APPWORLD_GROUP_GAMMA_ROUTE 只能是 0|1,实得 {route_raw!r}")
    group_gamma_route = route_raw == "1"
    if group_gamma_route and adaptive_spread:
        raise ValueError(
            "APPWORLD_GROUP_GAMMA_ROUTE=1 与 APPWORLD_ADAPTIVE_SPREAD=1 互斥"
            "（0.22a 单变量：全胜组不置零、只换折扣；叠加说不清归因）"
        )
    # 0.31 token 均摊开关。取值纪律同上：拼错静默落回默认 = 「token 臂其实是轮级臂」。
    spread_raw = os.environ.get("APPWORLD_TOKEN_SPREAD", "0")
    if spread_raw not in ("0", "1"):
        raise ValueError(f"APPWORLD_TOKEN_SPREAD 只能是 0|1,实得 {spread_raw!r}")
    token_spread = spread_raw == "1"
    if token_spread:
        if gamma != 1.0:
            raise ValueError(
                f"APPWORLD_TOKEN_SPREAD=1 要求 APPWORLD_GIGPO_GAMMA=1.0（实得 {gamma}）："
                "token 级均摊只在不打折的 reward-to-go 下有定义"
            )
        if group_gamma_route or adaptive_spread:
            raise ValueError(
                "APPWORLD_TOKEN_SPREAD=1 与 ADAPTIVE_SPREAD/GROUP_GAMMA_ROUTE 互斥（0.31 单变量：只换分辨率）"
            )
        if non_tensor_batch.get("outcome_score") is None:
            raise ValueError(
                "APPWORLD_TOKEN_SPREAD=1 但 non_tensor_batch 缺 outcome_score，"
                "无法把末轮 reward 拆成成败与进度两部分（fail closed）"
            )

    lm_arr = non_tensor_batch.get("loss_masked")
    valid = [not bool(lm_arr[i]) if lm_arr is not None else True for i in range(bsz)]
    n_masked = bsz - sum(valid)

    # P1-4（token 剂量）：response_mask 在此已是 actor update 真正消费的模型
    # token mask（tool 返回为 0，infra 行整行 0）。按原始 turn_end_pos 切段，折叠/截断
    # 后没有可更新 token 的轮明确记 0，不用生成长度近似替代。
    turn_end_arr = non_tensor_batch.get("turn_end_pos")
    extra_info_arr = non_tensor_batch.get("extra_info")
    update_token_rows: list[dict[str, Any]] = []
    update_model_tokens_total = 0
    if turn_end_arr is not None:
        for i in range(bsz):
            turn_ends = turn_end_arr[i] or []
            start = 0
            counts: list[int] = []
            for raw_end in turn_ends:
                end = min(max(int(raw_end), -1), resp_len - 1)
                if not valid[i] or end < start:
                    count = 0
                else:
                    count = int(response_mask[i, start : end + 1].sum().item())
                counts.append(count)
                start = max(start, end + 1)
            task_id = ""
            if extra_info_arr is not None and isinstance(extra_info_arr[i], dict):
                task_id = str(extra_info_arr[i].get("task_id", ""))
            update_model_tokens_total += sum(counts)
            update_token_rows.append(
                {
                    "row": i,
                    "uid": str(index[i]),
                    "task_id": task_id,
                    "tokens_by_turn": counts,
                }
            )

    with torch.no_grad():
        # ---------------- episode 层(GRPO 克隆 + mask 行排除组统计) ----------------
        # float64 求和(0.11c 审查补充1 实锤修复):步级散射后 float32 逐位求和的望远镜
        # 残差 ~1e-8,死组(同题同分)下被 (score−mean)/(std+eps) 的 0/0 型放大成
        # ~1e-2 级 episode 野信号(回归测试 test_dead_group_episode_adv_no_float32_blowup
        # 修前实测 0.0146)。float64 把顺序残差压到 ~1e-17;val 的 float32 表示误差由
        # 下面的近同分组保护兜底(仅 float64 仍泄漏 0.0092,实测)。
        scores = token_level_rewards.sum(dim=-1, dtype=torch.float64).clone()
        id2score = defaultdict(list)
        for i in range(bsz):
            if valid[i]:
                id2score[index[i]].append(scores[i])
        id2mean, id2std = {}, {}
        neardup_groups: set = set()
        for idx, vals in id2score.items():
            if len(vals) == 1:
                # 有意偏离 GRPO 的孤行约定(mean=0 → adv=原始分):此处孤行只在
                # infra mask 打掉组内其余轨迹时出现,给原始分会注入野信号 → 置 0
                id2mean[idx] = vals[0]
                id2std[idx] = torch.tensor(1.0)
            else:
                st = torch.stack(vals)
                # 近同分组保护(0.11c 审查补充1):组内极差低于 reward 物理分辨率
                # (断言粒度 ≥0.01,阈值留 3 个数量级余量)时,分差只能是 float32
                # 表示/求和噪声,(score−mean)/(std+eps) 是 0/0 型 → 优势置 0(死组同分语义)。
                if float(st.max() - st.min()) < 1e-5:
                    neardup_groups.add(idx)
                id2mean[idx] = torch.mean(st)
                id2std[idx] = torch.std(st)
        # P1-4：保护会把这些组的 episode 优势置零，因此必须在归零前留下原始读数。
        episode_neardup_score_range_max = 0.0
        episode_neardup_raw_adv_absmax = 0.0
        for idx in neardup_groups:
            st = torch.stack(id2score[idx])
            episode_neardup_score_range_max = max(
                episode_neardup_score_range_max, float(st.max() - st.min())
            )
            for score in st:
                if norm_by_std:
                    raw_adv = (score - id2mean[idx]) / (id2std[idx] + epsilon)
                else:
                    raw_adv = score - id2mean[idx]
                episode_neardup_raw_adv_absmax = max(
                    episode_neardup_raw_adv_absmax, float(raw_adv.abs())
                )
        ep = torch.zeros(bsz, dtype=torch.float32)
        for i in range(bsz):
            if not valid[i] or index[i] not in id2mean or index[i] in neardup_groups:
                continue
            if norm_by_std:
                ep[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                ep[i] = scores[i] - id2mean[index[i]]
        episode_adv = ep.unsqueeze(-1) * response_mask

        # 0.11d C2 构造级中继：全失败同题组中，若 shaping 总分已有物理可分辨差异，
        # episode 层必须真的产生非零优势；纯成败臂则应保持全零。这里直接在估计器
        # 消费的 score/adv 上计数，避免事后从 rounded rollout dump 近似重建。
        outcome_arr = non_tensor_batch.get("outcome_score")
        n_all_failure_groups = 0
        n_all_failure_shaping_diff_groups = 0
        n_all_failure_diff_groups_episode_nonzero = 0
        n_all_failure_episode_nonzero_rows = 0
        # 0.18a 新增（纯观测，不参与任何优势计算）：组结构的另一半。
        # 死组(n_all_failure_groups)一直在记，全胜组从来没记过，而 0.14a 把 27B 那次
        # 「摊步 − 终点 = −8.04」归因到**优势层长度偏置**时，关键证据正是全胜组占比 36-70%
        #  ——组内全对时成败信号消失，折扣 RTG 只剩「短=优」这条相关。4B 那边全胜组只有 0-4%
        # 所以同一机制被成败信号盖住了。要在训练中实时看见这个机制发动，就得有这条曲线：
        # 摊步臂的 C4 一旦往上飙，长度偏置就开始接管，属于报警而不是事后解剖。
        n_valid_groups = 0
        n_all_win_groups = 0
        # 0.19a：全对组的 uid 集合。这里只是把已经在算的 C4 判定顺手留个名单，
        # 判定口径与 n_all_win_groups **同一行代码**——报警曲线和干预用的必须是同一个判据，
        # 否则「C4 飙到 40%」和「置零了几组」会对不上号，读数时无从判断哪个是真的。
        all_win_uids: set = set()
        episode_nonzero_rows = sum(1 for i in range(bsz) if valid[i] and float(ep[i].abs()) > epsilon)
        if adaptive_spread and outcome_arr is None:
            # fail closed：开了自适应却没有 outcome_score，静默不置零 = 本臂退化成 0.18a 摊步臂，
            # 而 log 全绿、dump 也看不出来（0.11b 事故形状）。宁可当场炸训练。
            raise ValueError(
                "APPWORLD_ADAPTIVE_SPREAD=1 但 non_tensor_batch 缺 outcome_score，"
                "无法判定全对组（检查 aw_agent_loop 是否把 outcome_score 带进 non_tensor_batch）"
            )
        if group_gamma_route and outcome_arr is None:
            # 0.22a 同款 fail closed：缺 outcome 判不了全胜组 → 路由静默失效＝成本臂退化成普通摊步臂。
            raise ValueError(
                "APPWORLD_GROUP_GAMMA_ROUTE=1 但 non_tensor_batch 缺 outcome_score，"
                "无法判定全胜组（检查 aw_agent_loop 是否把 outcome_score 带进 non_tensor_batch）"
            )
        if outcome_arr is not None:
            grouped_rows: dict[Any, list[int]] = defaultdict(list)
            for i in range(bsz):
                if valid[i]:
                    grouped_rows[index[i]].append(i)
            for gkey, members in grouped_rows.items():
                if not members:
                    continue
                n_valid_groups += 1
                if all(float(outcome_arr[i]) != 0.0 for i in members):
                    n_all_win_groups += 1
                    all_win_uids.add(gkey)
                if any(float(outcome_arr[i]) != 0.0 for i in members):
                    continue
                n_all_failure_groups += 1
                nonzero_rows = sum(1 for i in members if float(ep[i].abs()) > epsilon)
                n_all_failure_episode_nonzero_rows += nonzero_rows
                group_scores = torch.stack([scores[i] for i in members])
                if float(group_scores.max() - group_scores.min()) >= 1e-5:
                    n_all_failure_shaping_diff_groups += 1
                    if nonzero_rows > 0:
                        n_all_failure_diff_groups_episode_nonzero += 1

        # 长度-优势相关（0.18a 新增，纯观测）：只在 episode 层算，样本 = 有效行。
        # 用 float 手算而不是引 scipy：估计器跑在训练主循环里，别为一条仪表加依赖。
        len_adv_pearson = 0.0
        _rows = [i for i in range(bsz) if valid[i]]
        if len(_rows) >= 3:
            _xs = [float(response_mask[i].sum()) for i in _rows]
            _ys = [float(ep[i]) for i in _rows]
            _mx = sum(_xs) / len(_xs)
            _my = sum(_ys) / len(_ys)
            _cov = sum((x - _mx) * (y - _my) for x, y in zip(_xs, _ys))
            _vx = sum((x - _mx) ** 2 for x in _xs)
            _vy = sum((y - _my) ** 2 for y in _ys)
            if _vx > 0 and _vy > 0:
                len_adv_pearson = round(_cov / ((_vx**0.5) * (_vy**0.5)), 6)

        # ---------------- step 层(Φ 锚点组内归一 turn 级 RTG) ----------------
        per_row, n_step_rows, n_mismatch = _rows_records(token_level_rewards, non_tensor_batch, valid)
        step_adv = torch.zeros_like(response_mask, dtype=torch.float32)
        anchor_group_sizes: list[int] = []
        unique_sizes: list[int] = []
        nonzero_var_groups = 0
        nonzero_var_groups_unique = 0
        n_multi_visit_records = 0
        real_anchor_groups = 0
        fallback = False
        n_records = 0
        n_none_anchor = 0
        # 0.22a：两个路由计数无论开关开没开都初始化（关着恒 0，仪表要能区分「关」和「开了没路由到」）
        n_gamma_routed_rows = 0
        n_gamma_full_rows = 0
        # 0.31：token 均摊仪表。ramp 张量单独留一份，读数时能反推「均摊前」的 step 层量级。
        n_token_spread_records = 0
        token_spread_ramp = torch.zeros_like(response_mask, dtype=torch.float32)

        if n_step_rows > 0:
            # RTG
            for i in range(bsz):
                recs = per_row[i]
                if not recs:
                    continue
                # 0.22a 按组换 γ：全胜组保满剂量 γ、其余组 γ=1.0。开关关着时 row_gamma
                # 恒等于 gamma（同一浮点值、同一条乘加路径）＝逐位不变，有对拍单测锁死。
                row_gamma = gamma
                if group_gamma_route and index[i] not in all_win_uids:
                    row_gamma = 1.0
                    n_gamma_routed_rows += 1
                elif group_gamma_route:
                    n_gamma_full_rows += 1
                running = 0.0
                for r in reversed(recs):
                    running = r["reward"] + row_gamma * running
                    r["rtg"] = running
                n_records += len(recs)
                n_none_anchor += sum(1 for r in recs if r["anchor"] is None)

            # anchor:按 (uid,进度锚点) 分组，稀疏时允许预注册 fallback。
            # task:显式按 uid 分组，不借用 fallback 路径伪装。
            groups: dict[Any, list[tuple[int, dict]]] = defaultdict(list)
            for i in range(bsz):
                recs = per_row[i]
                if not recs:
                    continue
                for k, r in enumerate(recs):
                    if group_mode == "task":
                        key = index[i]
                    elif r["anchor"] is None:
                        key = ("__none__", i, k)  # 独占组
                    else:
                        key = (index[i], r["anchor"])
                    groups[key].append((i, r))
            # 锚点组画像必须在 fallback 重建 groups **之前**定稿:否则 fallback 时
            # anchor_group_sizes/nonzero_var_groups 记的是 uid 全池、分母却是锚点组数
            # → nonzero_var_group_frac 可能 >1(审查 P2-b)。这三个指标恒表示「真锚点组」画像。
            real_groups = (
                list(groups.values())
                if group_mode == "task"
                else [v for kk, v in groups.items() if kk[0] != "__none__"]
            )
            real_sizes = [len(v) for v in real_groups]
            real_anchor_groups = len(real_sizes)
            anchor_group_sizes = list(real_sizes)
            # 独立轨迹口径(0.11c P0-4):每组去重 row 数;记录数−独立数 = 同轨迹驻留重复
            unique_sizes = [len({i for i, _ in v}) for v in real_groups]
            n_multi_visit_records = sum(len(v) - u for v, u in zip(real_groups, unique_sizes))
            for members, usize in zip(real_groups, unique_sizes):
                if len(members) > 1:
                    rtgs = torch.tensor([r["rtg"] for _, r in members], dtype=torch.float32)
                    if float(torch.std(rtgs)) > 0:
                        nonzero_var_groups += 1
                # P1-5：先把同一轨迹在本组内的驻留记录聚合，再比较不同轨迹。
                # 仅用 usize>=2 当门槛仍会把一条轨迹自身的轮间方差算进去。
                by_row: dict[int, list[float]] = defaultdict(list)
                for row_i, record in members:
                    by_row[row_i].append(record["rtg"])
                if len(by_row) >= 2:
                    row_means = torch.tensor(
                        [statistics.mean(values) for values in by_row.values()], dtype=torch.float32
                    )
                    if float(torch.std(row_means)) > 0:
                        nonzero_var_groups_unique += 1

            # fallback 判据(P0-4 修订):以独立轨迹数为主口径
            fallback = group_mode == "anchor" and (
                (not unique_sizes) or statistics.median(unique_sizes) < 2
            )
            if fallback:
                groups = defaultdict(list)
                for i in range(bsz):
                    recs = per_row[i]
                    if not recs:
                        continue
                    for r in recs:
                        groups[index[i]].append((i, r))

            for key, members in groups.items():
                rtgs = torch.tensor([r["rtg"] for _, r in members], dtype=torch.float32)
                if len(members) == 1:
                    mean = rtgs[0]  # GiGPO 孤组约定:mean=自身 → 优势 0
                    std = torch.tensor(1.0)
                else:
                    mean = torch.mean(rtgs)
                    std = torch.std(rtgs)
                for (i, r), rtg in zip(members, rtgs):
                    if mode == "mean_std_norm":
                        r["adv"] = float((rtg - mean) / (std + epsilon))
                    else:
                        r["adv"] = float(rtg - mean)

            # 广播:记录 k 覆盖 (p_{k-1}+1 .. p_k],首记录从 0 起
            spread_outcome_arr = non_tensor_batch.get("outcome_score") if token_spread else None
            for i in range(bsz):
                recs = per_row[i]
                if not recs:
                    continue
                start = 0
                for k, r in enumerate(recs):
                    step_adv[i, start : r["pos"] + 1] = r["adv"]
                    if token_spread:
                        # 0.31：末轮记录含成败，成败不摊；其余记录 reward 全是进度分。
                        r_prog = float(r["reward"])
                        if k == len(recs) - 1:
                            r_prog -= float(spread_outcome_arr[i])
                        span_mask = response_mask[i, start : r["pos"] + 1]
                        n_tok = int(span_mask.sum().item())
                        if r_prog != 0.0 and n_tok > 0:
                            tok_idx = torch.nonzero(span_mask, as_tuple=False).flatten() + start
                            ramp = torch.arange(n_tok, dtype=torch.float32) * (r_prog / n_tok)
                            step_adv[i, tok_idx] = step_adv[i, tok_idx] - ramp
                            token_spread_ramp[i, tok_idx] = ramp
                            n_token_spread_records += 1
                    start = r["pos"] + 1
            step_adv = step_adv * response_mask

        # ---------------- 0.19a 自适应摊步:全对组的 step 层整组置零 ----------------
        # 放在这里(RTG/分组/归一/广播/mask 全部走完)是刻意的,理由见模块 docstring。
        # 这段无论 n_step_rows 是否为 0 都要跑：仪表要能区分「没有全对组」和「有但没置零」。
        # 注意 all_win_uids 在 outcome_score 缺失时恒为空集(开关关着才允许走到这里),
        # 于是 aw_rows 空、other_rows = 全部有效行 —— 「非全胜组」在那种情况下读作
        # 「未知是否全胜」。开关开着时缺 outcome_score 已经在上面 fail closed 了。
        aw_rows = [i for i in range(bsz) if valid[i] and index[i] in all_win_uids]
        other_rows = [i for i in range(bsz) if valid[i] and index[i] not in all_win_uids]

        def _rows_rms_absmax(x: torch.Tensor, rows: list[int]) -> tuple[float, float]:
            """在给定行的**有效 token** 上算 (rms, absmax);行集为空或无有效 token → (0,0)。"""
            if not rows:
                return 0.0, 0.0
            sel = x[rows][response_mask[rows].bool()]
            if sel.numel() == 0:
                return 0.0, 0.0
            sel = sel.float()
            return float(torch.sqrt(torch.mean(sel**2))), float(sel.abs().max())

        step_adv_rms_all_win_pre, step_adv_absmax_all_win_pre = _rows_rms_absmax(step_adv, aw_rows)
        n_adaptive_zeroed_groups = 0
        n_adaptive_zeroed_rows = 0
        if adaptive_spread and aw_rows:
            for i in aw_rows:
                step_adv[i] = 0.0
            n_adaptive_zeroed_groups = len(all_win_uids)
            n_adaptive_zeroed_rows = len(aw_rows)
        step_adv_rms_all_win, step_adv_absmax_all_win = _rows_rms_absmax(step_adv, aw_rows)
        step_adv_rms_non_all_win, step_adv_absmax_non_all_win = _rows_rms_absmax(step_adv, other_rows)
        # 全对组的 **episode 层**读数(纯观测,永远不置零)。为什么必须有这条:0.19a 的目标写的是
        # 「让全对组回到真正的零梯度」,而 step 层抹零只做了一半——TERMINAL=keep 下
        # episode 分 = outcome + coef·(Φ_T−Φ_0),组内全对时 Φ_T 若逐条相同,近同分保护会把它
        # 归零(=真零梯度);若不同(例如 TGC=1 但断言集合不一样),episode 层仍有残余优势。
        # 这条读数就是「零梯度到底成立没成立」的唯一证据,不测就只能靠推断。
        episode_adv_rms_all_win, episode_adv_absmax_all_win = _rows_rms_absmax(episode_adv, aw_rows)

        advantages = episode_adv + w * step_adv
        # mask 行优势强制清零(其 response_mask 本已全 0,双保险)
        for i in range(bsz):
            if not valid[i]:
                advantages[i] = 0.0

        # ---- 量纲与泄漏观测(0.11c P0-2):episode 层除 std、step 层 mean-only 的
        # 混合口径下,w=1.0 的名义值没有稳定「1:1」语义,实际混合比按两层 RMS 落盘;
        # 组内近同分时的 episode 泄漏绝对值最大者落盘为实锤观测。 ----
        valid_mask_rows = torch.tensor(valid, dtype=torch.bool)
        tok_valid = response_mask[valid_mask_rows].bool() if int(valid_mask_rows.sum()) else torch.zeros(0, dtype=torch.bool)

        def _rms(x: torch.Tensor) -> float:
            sel = x[valid_mask_rows][tok_valid] if tok_valid.numel() else torch.zeros(0)
            return float(torch.sqrt(torch.mean(sel.float() ** 2))) if sel.numel() else 0.0

        episode_adv_rms = _rms(episode_adv)
        step_adv_rms = _rms(step_adv)
        episode_neardup_leak_absmax = 0.0
        for idx_key in neardup_groups:
            for i in range(bsz):
                if valid[i] and index[i] == idx_key:
                    episode_neardup_leak_absmax = max(episode_neardup_leak_absmax, float(ep[i].abs()))

    # 0.31：均摊前/后 step 层在模型 token 上的 Pearson。关开关时 ramp 全 0 → 恒 1.0；
    # 零方差（本批没有任何 step 信号）也记 1.0，表示「没东西可比」而非「完全不相关」。
    _ts_m = response_mask.bool()
    _ts_post = step_adv[_ts_m].float()
    _ts_pre = (step_adv + token_spread_ramp * response_mask)[_ts_m].float()
    _token_spread_pearson = 1.0
    if _ts_post.numel() > 1 and float(_ts_post.std()) > 0 and float(_ts_pre.std()) > 0:
        _token_spread_pearson = float(torch.corrcoef(torch.stack([_ts_pre, _ts_post]))[0, 1])

    global _CALL_SEQ
    _CALL_SEQ += 1
    LAST_STATS.clear()
    LAST_STATS.update(
        {
            "estimator": "appworld_phi_gigpo",
            # ---- 对账键(0.11c P0-5 + 复审二 P0-B;判决脚本按
            #      (run_id, attempt_id, global_step) keyed join,fail closed) ----
            "run_id": os.environ.get("APPWORLD_RUN_ID", ""),
            "attempt_id": int(os.environ.get("APPWORLD_RUN_ATTEMPT", "0") or 0),
            "call_seq": _CALL_SEQ,
            "global_step": (meta_info or {}).get("global_steps", -1),
            "ts": round(time.time(), 3),
            "mode": mode,
            "group_mode": group_mode,
            "gamma": gamma,
            "w": w,
            "norm_adv_by_std_in_grpo": norm_by_std,
            "bsz": bsz,
            "n_masked_rows": n_masked,
            "n_step_rows": n_step_rows,
            "n_records": n_records,
            "n_none_anchor_records": n_none_anchor,
            "n_real_anchor_groups": real_anchor_groups,
            "anchor_group_size_median": (statistics.median(anchor_group_sizes) if anchor_group_sizes else 0),
            "anchor_group_size_max": (max(anchor_group_sizes) if anchor_group_sizes else 0),
            "nonzero_var_group_frac": (nonzero_var_groups / real_anchor_groups if real_anchor_groups else 0.0),
            # ---- 独立轨迹口径(0.11c P0-4;fallback 主判据 + C1-② 主口径) ----
            "anchor_group_unique_median": (statistics.median(unique_sizes) if unique_sizes else 0),
            "anchor_group_unique_max": (max(unique_sizes) if unique_sizes else 0),
            "n_multi_visit_records": n_multi_visit_records,
            "nonzero_var_group_frac_unique": (nonzero_var_groups_unique / real_anchor_groups if real_anchor_groups else 0.0),
            # ---- 量纲/泄漏(0.11c P0-2) ----
            "episode_adv_rms": round(episode_adv_rms, 6),
            # 0.19a 起 step_adv_rms 是**置零之后**的值(=真正进优势的那份混合比)。
            # 跨臂比这一项时要记着:自适应臂的它天然比同批锚点臂小,少掉的正是全对组那部分,
            # 不是「步级信号变弱了」。想看置零前的量级用 step_adv_rms_all_win_pre。
            "step_adv_rms": round(step_adv_rms, 6),
            "n_episode_neardup_groups": len(neardup_groups),
            "episode_neardup_score_range_max": episode_neardup_score_range_max,
            "episode_neardup_raw_adv_absmax": episode_neardup_raw_adv_absmax,
            "episode_neardup_leak_absmax": round(episode_neardup_leak_absmax, 6),
            "episode_nonzero_rows": episode_nonzero_rows,
            # ---- 0.18a 新增：组结构 C4 与长度偏置仪表（纯观测）----
            # ---- 0.19a 自适应摊步（开关 + 指纹；smoke/读数的自动核对就查这几个）----
            # 判据：adaptive_spread=true 时 step_adv_absmax_all_win 必须恒为 0，
            # 且 n_all_win_groups>0 而 step_adv_absmax_all_win_pre>0 时说明真的抹掉了东西
            # （pre 全 0 表示这批全对组本来就没轮级信号，属于「无事可做」而非「没生效」）。
            "adaptive_spread": adaptive_spread,
            "n_adaptive_zeroed_groups": n_adaptive_zeroed_groups,
            "n_adaptive_zeroed_rows": n_adaptive_zeroed_rows,
            # 0.22a：路由开关与两侧行数。判「这个 run 到底路由没路由」只认这三个键 +
            # launch_params（目录名不作数，0.16a 教训）。
            "group_gamma_route": group_gamma_route,
            "n_gamma_routed_rows": n_gamma_routed_rows,
            "n_gamma_full_rows": n_gamma_full_rows,
            # 0.31：判「这个 run 到底摊没摊」只认这几个键 + launch_params（目录名不作数）。
            # 关开关时 n_token_spread_records 与 ramp_rms 恒为 0；开着时 n>0 且 ramp_rms>0
            # 才是「真摊了」；开着但 n=0 表示本批没有任何推进轮，属于无事可做不是没生效。
            "token_spread": token_spread,
            "n_token_spread_records": n_token_spread_records,
            "token_spread_ramp_rms": round(_rms(token_spread_ramp), 6),
            "step_adv_rms_pre_token_spread": round(_rms(step_adv + token_spread_ramp * response_mask), 6),
            # 均摊前后 step 层逐 token Pearson（=0.18x 那种「离线相关」直接从 run 里读，不用另跑筛子）。
            "token_spread_pearson_pre_post": round(_token_spread_pearson, 6),
            "step_adv_rms_all_win_pre": round(step_adv_rms_all_win_pre, 6),
            "step_adv_absmax_all_win_pre": round(step_adv_absmax_all_win_pre, 6),
            "step_adv_rms_all_win": round(step_adv_rms_all_win, 6),
            "step_adv_absmax_all_win": round(step_adv_absmax_all_win, 6),
            "step_adv_rms_non_all_win": round(step_adv_rms_non_all_win, 6),
            "step_adv_absmax_non_all_win": round(step_adv_absmax_non_all_win, 6),
            # 「全对组是否真的零梯度」的判据（episode 层永不置零，这条纯观测）：
            # 自适应臂里它若恒为 0，全对组就是真零梯度；若 >0，说明还有 shaping 残余优势，
            # 结论只能写「轮级毒性已拔掉」，不能写「全对组零梯度」。
            "episode_adv_rms_all_win": round(episode_adv_rms_all_win, 6),
            "episode_adv_absmax_all_win": round(episode_adv_absmax_all_win, 6),
            "n_valid_groups": n_valid_groups,
            "n_all_win_groups": n_all_win_groups,
            "all_win_group_frac": (n_all_win_groups / n_valid_groups if n_valid_groups else 0.0),
            "all_failure_group_frac": (n_all_failure_groups / n_valid_groups if n_valid_groups else 0.0),
            # 响应长度与 episode 层优势的 Pearson 相关。0.14a 的长度偏置就是靠它定的性：
            # 全胜组里成败信号消失后，折扣 RTG 退化成「短=优」，这个相关会往 −1 走。
            "len_episode_adv_pearson": len_adv_pearson,
            "n_all_failure_groups": n_all_failure_groups,
            "n_all_failure_shaping_diff_groups": n_all_failure_shaping_diff_groups,
            "n_all_failure_diff_groups_episode_nonzero": n_all_failure_diff_groups_episode_nonzero,
            "n_all_failure_episode_nonzero_rows": n_all_failure_episode_nonzero_rows,
            "fallback_turn_rtg": fallback,
            "n_val_mismatch_rows": n_mismatch,
            "update_model_tokens_total": update_model_tokens_total,
            "update_token_rows": update_token_rows,
        }
    )
    # D1' 机制指标是判据来源,必须可靠落地(审查 P2-a):
    # ① warning 级进 train.log(root logger 若压到 WARNING 仍在);② 追加落盘 JSONL
    # (LAST_STATS 是 driver 进程内 module 全局,判决脚本跨进程读不到)。
    logger.warning("appworld_phi_gigpo stats: %s", LAST_STATS)
    dump_dir = os.environ.get("APPWORLD_ROLLOUT_DUMP")
    if dump_dir:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, "gigpo_stats.jsonl"), "a") as f:
                f.write(json.dumps(LAST_STATS, ensure_ascii=False) + "\n")
        except OSError as e:  # 落盘失败不炸训练,日志已有兜底
            logger.warning("appworld_phi_gigpo: stats 落盘失败: %s", e)
    if n_mismatch:
        logger.warning(
            "appworld_phi_gigpo: %d 行 token_level_rewards 与注入值失配"
            "(散射块是否在运行时那棵 verl 里?检查 verl/experimental/agent_loop/agent_loop.py)",
            n_mismatch,
        )
    return advantages, advantages


# 注册(driver 进程经 VERL_IMPORT_MODULES=appworld_tool.aw_gigpo 导入本模块时生效);
# 纯单测环境无 verl 时跳过,核心函数仍可直接测。
try:
    from verl.trainer.ppo.core_algos import register_adv_est

    register_adv_est("appworld_phi_gigpo")(compute_appworld_phi_gigpo_advantage)
except ImportError:  # pragma: no cover
    logger.info("verl 不可导入,appworld_phi_gigpo 未注册(单测模式)")

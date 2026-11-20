"""Multi-turn AppWorld ReAct loop for verl.

The initial prompt is provided in raw_prompt. Model responses and tool
observations are appended to the conversation; response_mask selects only
model-generated tokens. Environment evaluations provide per-turn progress,
with reward values and token positions passed to the training framework.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from appworld_tool import aw_reward
from appworld_tool.aw_client import AppWorldEnvClient, acquire_port, get_http_session, get_port_pool

logger = logging.getLogger(__file__)

CODE_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)


def parse_action(text: str) -> tuple[str, int]:
    """与 0705_sample_rollouts_react.py 的 parse_action 逐字一致。"""
    blocks = CODE_RE.findall(text)
    if not blocks:
        return text[-100:], 0
    return blocks[-1], 1


class AppWorldReactAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_turns = int(os.environ.get("APPWORLD_MAX_TURNS", "50"))
        self.per_turn_tokens = int(os.environ.get("APPWORLD_PER_TURN_TOKENS", "512"))
        self.ctx_budget = int(os.environ.get("APPWORLD_CTX_BUDGET", "32768"))
        self.eval_every_turn = os.environ.get("APPWORLD_EVAL_EVERY_TURN", "1") == "1"
        self._pool = get_port_pool()

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        t0 = time.monotonic()
        extra_info = kwargs.get("extra_info", {}) or {}
        task_id = extra_info.get("task_id") or kwargs.get("task_id")
        assert task_id, "parquet extra_info 必须带 task_id"
        uid = str(kwargs.get("uid", ""))
        step = kwargs.get("global_steps", -1)
        # Read split metadata from the dataset when the trainer omits __validate__.
        split = extra_info.get("split") or ("val" if kwargs.get("__validate__", False) else "train")

        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)
        init_prompt_len = len(prompt_ids)

        sp = {**sampling_params, "max_tokens": self.per_turn_tokens}
        request_id = uuid4().hex

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        turns: list[dict] = []
        phi_seq: list[float] = []  # legacy 0.10b:仅成功时 append,与轮次不对齐(轨迹级只取首尾)
        eval_wall: list[float] = []
        infra_errors: list[str] = []
        # ---- 0.11b 轮次对齐结构:下标 t 即轮次(0 = 初始 evaluate),缺测填 None ----
        phi_by_turn: list[float | None] = []
        anchor_by_turn: list[str | None] = []
        passed_idx_by_turn: list[list[int] | None] = []
        turn_end_pos: list[int] = []  # 每轮末 model token 在 response 帧的下标(len = 轮数)
        req_index: dict[str, int] = {}  # 净断言 requirement → 稳定下标(逐断言漂移分析用)
        completed = False
        aborted_over_ctx = False
        phi0_missing = False
        final_eval: dict = {}
        metrics: dict = {}
        acquire_wait_s = 0.0

        def _req_idx(reqs: list[str]) -> list[int]:
            for r in reqs:
                if r not in req_index:
                    req_index[r] = len(req_index)
            return sorted(req_index[r] for r in reqs)

        async def eval_once(client: AppWorldEnvClient) -> dict | None:
            """一次 evaluate。失败 = 该轮 Φ 缺测,不炸(返回 None)。"""
            if not self.eval_every_turn:
                return None
            te = time.monotonic()
            try:
                ev = await client.evaluate()
                eval_wall.append(time.monotonic() - te)
                return ev
            except Exception as e:  # noqa: BLE001
                infra_errors.append(f"turn_eval: {str(e)[:120]}")
                return None

        def record_phi(ev: dict | None) -> None:
            """legacy phi_seq(逐字保持 0.10b:仅成功时 append)+ 轮次对齐的 phi/anchor/断言集。"""
            if ev is not None:
                phi_seq.append(aw_reward.phi_of(ev))
                passed, all_net = aw_reward.net_requirements(ev)
                _req_idx(all_net)  # 先登记全集,保证下标稳定(通过集是它的子集)
                phi_by_turn.append(aw_reward.phi_of(ev))
                anchor_by_turn.append(aw_reward.anchor_of(passed))
                passed_idx_by_turn.append(_req_idx(passed))
            else:
                phi_by_turn.append(None)
                anchor_by_turn.append(None)
                passed_idx_by_turn.append(None)

        # Retry initialization with another lease if a server is unavailable.
        exp_name_base = f"aw10b_{split}_s{step}_{uid[:8]}_{request_id[:8]}"
        init_tries = int(os.environ.get("APPWORLD_INIT_PORT_TRIES", "6"))
        client = None
        for _try in range(init_tries):
            acquire_t0 = time.monotonic()
            port = await acquire_port(self._pool)
            acquire_wait_s += time.monotonic() - acquire_t0
            c = AppWorldEnvClient(port, get_http_session())
            try:
                await c.initialize(task_id, exp_name_base)
                client = c
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("initialize 失败(port %s, 第 %s 次): %s", port, _try + 1, str(e)[:150])
                try:
                    await self._pool.release.remote(port, bad=True)
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(5 * (_try + 1))
        if client is None:
            raise RuntimeError(f"initialize 连续 {init_tries} 个端口失败(task {task_id})")
        try:
            record_phi(await eval_once(client))  # phi_by_turn[0] = Φ_0
            phi0_missing = self.eval_every_turn and not phi_seq

            for turn in range(1, self.max_turns + 1):
                # 上下文预算:下一轮生成 512 放不下 → over_ctx 收尾(照常终局 evaluate)
                if len(prompt_ids) + self.per_turn_tokens > self.ctx_budget or (
                    len(response_mask) + self.per_turn_tokens >= self.response_length
                ):
                    aborted_over_ctx = True
                    break

                out: TokenOutput = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sp
                )
                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
                response_text = self.tokenizer.decode(out.token_ids, skip_special_tokens=True)
                prompt_ids = prompt_ids + out.token_ids
                response_mask += [1] * len(out.token_ids)
                # 该轮末 model token 的下标(response 帧,与 response_ids 同序);空生成兜底夹到 0
                turn_end_pos.append(max(0, len(response_mask) - 1))
                if out.log_probs:
                    response_logprobs += out.log_probs

                code, valid = parse_action(response_text)
                try:
                    env_out = await client.execute(code)
                except Exception as e:  # noqa: BLE001 — server 卡死/连接断:收尾,终局 evaluate 仍会尝试
                    logger.warning("execute 失败(task %s turn %s): %s", task_id, turn, e)
                    infra_errors.append(f"execute: {str(e)[:120]}")
                    record_phi(None)  # 该轮 Φ 缺测:必须占位,否则 phi_by_turn 与轮次错位
                    turns.append({"turn": turn, "valid": valid, "env_error": True})
                    break

                record_phi(await eval_once(client))
                turns.append({"turn": turn, "valid": valid, "resp_tokens": len(out.token_ids)})

                try:
                    if await client.task_completed():
                        completed = True
                        break
                except Exception as e:  # noqa: BLE001 — 视为未完成,继续
                    infra_errors.append(f"task_completed: {str(e)[:120]}")

                if len(env_out) > 150_000:
                    # 病态巨型输出(实测一例 27 万 token:模型打印整库):必然超预算,
                    # 直接 over_ctx 收尾,跳过模板化(防框架 left-truncating 告警误报)
                    aborted_over_ctx = True
                    break
                user_msg = {"role": "user", "content": f"Output:\n```\n{env_out}\n```"}
                delta_ids = await self.apply_chat_template([user_msg], remove_system_prompt=True)
                if len(response_mask) + len(delta_ids) + self.per_turn_tokens >= self.response_length:
                    aborted_over_ctx = True
                    break
                prompt_ids = prompt_ids + delta_ids
                response_mask += [0] * len(delta_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(delta_ids)

            # 终局 evaluate:即使 execute 超时(server 标坏)也短超时试一次——未知终态
            try:
                final_eval = await client.evaluate(timeout_s=60)
            except Exception as e:  # noqa: BLE001
                infra_errors.append(f"final_eval: {str(e)[:120]}")
                client.suspect_bad = True
        finally:
            try:
                await client.close()
            finally:
                try:
                    await self._pool.release.remote(port, bad=client.suspect_bad)
                except Exception as e:  # noqa: BLE001 — release 失败只记日志,不吞掉主异常
                    logger.error("release port %s 失败: %s", port, e)

        infra_error = not final_eval or any(x.startswith(("execute", "final_eval")) for x in infra_errors)
        success = bool(final_eval.get("success", False))
        net_pass, net_n = aw_reward.net_counts(final_eval)
        final_phi = net_pass / net_n if net_n > 0 else 0.0
        if final_eval and self.eval_every_turn:
            # 终局 evaluate 为权威 Φ_T:替换末位(最后一轮 eval 与终局同状态),不重复 append
            if phi_seq:
                phi_seq[-1] = final_phi
            else:
                phi_seq.append(final_phi)
        scores = aw_reward.compute_reward(1.0 if success else 0.0, phi_seq or [0.0])

        # 0.11b:终局 evaluate 为权威 Φ_T → 写回**最后一轮**(轮次对齐;legacy 写的是"最后一个非缺测")
        if final_eval and self.eval_every_turn and len(phi_by_turn) >= 2:
            phi_by_turn[-1] = final_phi
            fin_passed, fin_all = aw_reward.net_requirements(final_eval)
            _req_idx(fin_all)
            anchor_by_turn[-1] = aw_reward.anchor_of(fin_passed)
            passed_idx_by_turn[-1] = _req_idx(fin_passed)

        outcome = 1.0 if success else 0.0
        step_pos, step_val = self._build_step_injection(
            outcome, phi_by_turn, turn_end_pos, len(response_mask), task_id
        )
        step_requested = aw_reward.step_credit_enabled() and self.eval_every_turn
        step_injection_degraded = bool(step_requested and not step_pos)
        step_episode_score = aw_reward.compute_step_episode_score(
            outcome, phi_by_turn, aw_reward.shaping_coef()
        )

        loss_masked = bool(os.environ.get("APPWORLD_INFRA_MASK", "0") == "1" and infra_error)
        if loss_masked:
            # 整条轨迹不出 loss;agg_loss 三模式均显式排除全掩码序列(core_algos.py:1174/1189)
            # 遗留:其 score 仍进 GRPO 组均值 → 须由 §3b 估计器排除出组统计
            response_mask = [0] * len(response_mask)

        response_ids = prompt_ids[init_prompt_len:]
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[:init_prompt_len],
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=len(turns) * 2 + 1,
            metrics=metrics,
        )
        # 步级开启时，正常散射和退化标量必须共用同一条总账公式。尤其 0.11d
        # APPWORLD_SHAPING=0，不能用 legacy final_score 回退，否则 keep 加分会静默消失。
        output.reward_score = step_episode_score if step_requested else scores["final_score"]
        output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "outcome_score": scores["outcome_score"],
                "shaping": scores["shaping"],
                "final_score": scores["final_score"],
                # ---- 0.11b:供 _postprocess 散射 + §3b 估计器取锚点/分组 ----
                "step_reward_pos": step_pos,
                "step_reward_val": step_val,
                "step_injection_degraded": step_injection_degraded,
                "anchor_by_turn": anchor_by_turn,
                "turn_end_pos": turn_end_pos,
                "infra_error": infra_error,
                "loss_masked": loss_masked,
            }
        )

        dump_dir = os.environ.get("APPWORLD_ROLLOUT_DUMP")
        if dump_dir:
            rec = {
                "task_id": task_id,
                "uid": uid,
                "step": step,
                "split": split,
                "request_id": request_id,
                "port": port,
                "num_turns": len(turns),
                "completed": completed,
                "censored": not completed,
                "aborted_over_ctx": aborted_over_ctx,
                "infra_error": infra_error,
                "infra_error_detail": infra_errors[:5],
                "phi0_missing": phi0_missing,
                "success": success,
                "net_pass": net_pass,
                "net_n": net_n,
                "invalid_turns": sum(1 for t in turns if not t["valid"]),
                "phi_seq": [round(p, 4) for p in phi_seq],
                "eval_wall_p50": round(sorted(eval_wall)[len(eval_wall) // 2], 3) if eval_wall else None,
                "acquire_wait_s": round(acquire_wait_s, 3),
                "response_len": len(response_mask),
                "truncated_out": len(response_mask) > self.response_length,
                **scores,
                # ---- 0.11b 逐断言 / 步级审计 ----
                "assert_ids": [r for r, _ in sorted(req_index.items(), key=lambda kv: kv[1])],
                "passed_idx_by_turn": passed_idx_by_turn,
                "anchor_by_turn": anchor_by_turn,
                "phi_by_turn": [None if p is None else round(p, 4) for p in phi_by_turn],
                "turn_end_pos": turn_end_pos,
                "step_reward_val": [round(v, 6) for v in step_val],
                # ---- 0.11c 审查补充2:step 系数无歧义落盘。legacy `shaping_coef` 在
                # APPWORLD_SHAPING=0 时恒记 0,但步级注入读的是 APPWORLD_SHAPING_COEF
                # ——两者语义分裂,判决脚本禁用 legacy 字段推导步级公式。 ----
                "step_credit_enabled": aw_reward.step_credit_enabled(),
                "step_shaping_coef": aw_reward.shaping_coef() if aw_reward.step_credit_enabled() else 0.0,
                "step_terminal_mode": aw_reward.step_terminal_mode(),
                "group_mode": os.environ.get("APPWORLD_GIGPO_GROUP", "anchor"),
                "injected_reward_sum": round(sum(step_val), 6) if step_val else None,
                "step_episode_score": round(step_episode_score, 6) if step_requested else None,
                "step_injection_degraded": step_injection_degraded,
                "loss_masked": loss_masked,
                "wall_s": round(time.monotonic() - t0, 1),
            }
            self._append_dump(dump_dir, rec)
        return output

    def _build_step_injection(
        self,
        outcome: float,
        phi_by_turn: list[float | None],
        turn_end_pos: list[int],
        resp_total: int,
        task_id: str,
    ) -> tuple[list[int], list[float]]:
        """步级 reward 的稀疏散射表 (positions, values)。关闭/退化 → ([], []) 由调用方回退标量。

        截断轮(位置越过 response_length)的 reward 折叠进最后一个有效 token，保住当前
        terminal mode 对应的望远镜和；这只是公式一致性，不是行为层防刷担保。
        """
        if not (aw_reward.step_credit_enabled() and self.eval_every_turn):
            return [], []
        T = len(phi_by_turn) - 1
        if T <= 0 or len(turn_end_pos) != T:
            logger.warning(
                "步级注入退化为标量(task %s):turn_end_pos=%d 应等于 T=%d", task_id, len(turn_end_pos), T
            )
            return [], []
        resp_eff = min(resp_total, self.response_length)
        if resp_eff <= 0:
            return [], []
        last_valid = resp_eff - 1
        rewards = aw_reward.compute_step_rewards(outcome, phi_by_turn, aw_reward.shaping_coef())
        folded: dict[int, float] = {}
        for p, v in zip(turn_end_pos, rewards):
            p = min(p, last_valid)
            folded[p] = folded.get(p, 0.0) + v
        pos = sorted(folded)
        return pos, [folded[p] for p in pos]

    @staticmethod
    def _append_dump(dump_dir: str, rec: dict) -> None:
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"dump_pid{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

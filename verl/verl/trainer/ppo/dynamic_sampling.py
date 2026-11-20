"""DAPO group filtering and accumulation for the training loop.

Groups are selected by score variance, excluding infrastructure-masked
trajectories. Accepted groups are accumulated into a training batch, with
explicit handling of partial batches and exhausted generation budgets.
Hook checks detect missing trainer integration before training proceeds.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from verl import DataProto


class DynamicSamplingExhaustedError(RuntimeError):
    """生成上限内没有得到任何可训练题组。"""


class DynamicSamplingHookError(RuntimeError):
    """配置开了动态采样，但跑的那棵树上挂接点缺失/不全 —— 必须炸，不许静默跑成普通 GRPO。"""


# 挂接点标记：ray_trainer.py 里每一处挂接都带一条形如 `DYNAMIC_SAMPLING_HOOK::<名字>` 的注释，
# 这里按「哪个方法里必须出现哪些标记」列清单。检查读的是 `inspect.getsource`，
# 也就是**真正被 import 进来的那份源码**，所以改错树 / 只改了一半都会被抓住。
HOOK_MARKERS: dict[str, tuple[str, ...]] = {
    "fit": (
        "DYNAMIC_SAMPLING_HOOK::setup",  # 构造 sampler + 打横幅 + 自检
        "DYNAMIC_SAMPLING_HOOK::loop_head",  # metrics/timing 跨补采批次不清零
        "DYNAMIC_SAMPLING_HOOK::gen_timing",  # 生成耗时累加（补采会生成多次）
        "DYNAMIC_SAMPLING_HOOK::filter",  # 真正的丢组/补采/闭合
        "DYNAMIC_SAMPLING_HOOK::loop_tail",  # 一步结束后复位 in-progress 标志
    ),
    "_update_actor": (
        "DYNAMIC_SAMPLING_HOOK::mini_batch",  # 触顶部分批次时 mini_batch 必须跟着缩
    ),
}


def _resolve_trainer_class(trainer) -> type:
    return trainer if isinstance(trainer, type) else type(trainer)


def assert_trainer_hooks_installed(trainer) -> dict[str, str]:
    """确认 DAPO 挂接点确实存在于**正在运行的**训练器源码里，缺一处就抛。

    Args:
        trainer: `RayPPOTrainer` 实例或类。

    Returns:
        dict: 诊断信息（本模块与训练器模块各自解析到的文件路径），供日志留痕。

    Raises:
        DynamicSamplingHookError: 任一挂接点标记缺失。
    """
    trainer_cls = _resolve_trainer_class(trainer)
    missing: list[str] = []
    for method_name, markers in HOOK_MARKERS.items():
        method = getattr(trainer_cls, method_name, None)
        if method is None:
            missing.append(f"{trainer_cls.__name__}.{method_name}（方法不存在）")
            continue
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError) as exc:  # pragma: no cover - 只在源码不可读时触发
            raise DynamicSamplingHookError(
                f"无法读取 {trainer_cls.__name__}.{method_name} 的源码做挂接自检：{exc}"
            ) from exc
        for marker in markers:
            if marker not in source:
                missing.append(f"{trainer_cls.__name__}.{method_name} 缺标记 {marker}")

    paths = {
        "dynamic_sampling": __file__,
        "trainer_module": inspect.getfile(trainer_cls),
    }
    if missing:
        raise DynamicSamplingHookError(
            "DAPO 动态采样已在配置里开启（algorithm.filter_groups.enable=True），"
            "但跑的这棵树上挂接点不全 —— 继续跑会静默退化成普通 GRPO（0.11b 同型事故）。\n"
            f"  dynamic_sampling.py = {paths['dynamic_sampling']}\n"
            f"  ray_trainer.py      = {paths['trainer_module']}\n"
            "  缺失项：\n    " + "\n    ".join(missing)
        )
    return paths


def format_active_banner(sampler: DynamicGroupSampler, paths: dict[str, str]) -> str:
    """启动时那条「不可能看漏」的横幅。丢组这件事不许只靠配置文件自证。"""
    bar = "=" * 78
    return "\n".join(
        [
            bar,
            "[DYNAMIC_SAMPLING] ***** DYNAMIC_SAMPLING ACTIVE (DAPO 丢同分组 + 补采) *****",
            f"[DYNAMIC_SAMPLING]   metric              = {sampler.metric}",
            f"[DYNAMIC_SAMPLING]   target_num_prompts  = {sampler.target_num_prompts}",
            f"[DYNAMIC_SAMPLING]   rollout_n           = {sampler.rollout_n}",
            f"[DYNAMIC_SAMPLING]   max_num_gen_batches = {sampler.max_num_gen_batches}",
            f"[DYNAMIC_SAMPLING]   module              = {paths['dynamic_sampling']}",
            f"[DYNAMIC_SAMPLING]   trainer             = {paths['trainer_module']}",
            "[DYNAMIC_SAMPLING]   判据 = 组内 metric 方差为 0 即丢（**全胜组和死组一起丢**）",
            bar,
        ]
    )


def build_sampler_from_config(config) -> DynamicGroupSampler | None:
    """按 `config.algorithm.filter_groups` 造 sampler；没开就返回 None。

    fail-closed 的几条都在这里：v1 专属旋钮被 v0 忽略、共置 RM、REMAX 额外基线。
    """
    filter_groups_config = config.algorithm.get("filter_groups", None)
    if not filter_groups_config or not filter_groups_config.get("enable", False):
        return None

    # `max_inflight_gen_batches` 只有 v1 的 replay_buffer 认；v0 这条同步路径完全忽略它。
    # 传了非默认值却被忽略＝又一个静默口子，直接拒。
    max_inflight = filter_groups_config.get("max_inflight_gen_batches", 1)
    if max_inflight not in (None, 1):
        raise DynamicSamplingHookError(
            f"algorithm.filter_groups.max_inflight_gen_batches={max_inflight} 只有 v1 训练器认，"
            "v0 同步路径会完全忽略它（传了等于没传）。要跑并发补采请另行设计，别靠这个旋钮。"
        )
    return DynamicGroupSampler(
        target_num_prompts=config.data.train_batch_size,
        rollout_n=config.actor_rollout_ref.rollout.n,
        metric=filter_groups_config.get("metric"),
        max_num_gen_batches=filter_groups_config.get("max_num_gen_batches"),
    )


@dataclass(frozen=True)
class DynamicSamplingStats:
    generation_batches: int
    generated_prompts: int
    kept_prompts: int
    dropped_prompts: int
    target_prompts: int
    max_hit: bool

    @property
    def drop_rate(self) -> float:
        return self.dropped_prompts / self.generated_prompts

    @property
    def generation_amplification(self) -> float:
        return self.generated_prompts / self.target_prompts


@dataclass(frozen=True)
class DynamicSamplingResult:
    ready: bool
    batch: DataProto | None
    stats: DynamicSamplingStats
    partial: bool = False


class DynamicGroupSampler:
    """过滤同分题组，并跨生成批次累积有梯度的完整题组。"""

    def __init__(
        self,
        *,
        target_num_prompts: int,
        rollout_n: int,
        metric: str,
        max_num_gen_batches: int,
    ) -> None:
        if target_num_prompts <= 0:
            raise ValueError("target_num_prompts 必须大于 0")
        if rollout_n <= 1:
            raise ValueError("rollout_n 必须大于 1，否则无法判断题组方差")
        if not metric:
            raise ValueError("metric 不能为空")
        if max_num_gen_batches <= 0:
            raise ValueError("max_num_gen_batches 必须大于 0")
        self.target_num_prompts = target_num_prompts
        self.rollout_n = rollout_n
        self.metric = metric
        self.max_num_gen_batches = max_num_gen_batches
        self._kept_batches: list[DataProto] = []
        self._generation_batches = 0
        self._generated_prompts = 0
        self._kept_prompts = 0
        self._dropped_prompts = 0

    def _stats(self, *, max_hit: bool = False) -> DynamicSamplingStats:
        return DynamicSamplingStats(
            generation_batches=self._generation_batches,
            generated_prompts=self._generated_prompts,
            kept_prompts=self._kept_prompts,
            dropped_prompts=self._dropped_prompts,
            target_prompts=self.target_num_prompts,
            max_hit=max_hit,
        )

    def _reset(self) -> None:
        self._kept_batches = []
        self._generation_batches = 0
        self._generated_prompts = 0
        self._kept_prompts = 0
        self._dropped_prompts = 0

    def add(self, batch: DataProto) -> DynamicSamplingResult:
        uids = batch.non_tensor_batch["uid"]
        if self.metric not in batch.non_tensor_batch:
            # 0.19c：metric 字段进不了 non_tensor_batch 是「配置看着对、其实读不到分」的静默口子，
            # 报错要把现有键列出来，别只抛一个裸 KeyError。
            raise DynamicSamplingHookError(
                f"动态采样的 metric='{self.metric}' 不在 non_tensor_batch 里；"
                f"现有键={sorted(batch.non_tensor_batch.keys())}。"
                "AppWorld 这条链上它由 aw_agent_loop 的 extra_fields 带出来，检查 reward_extra_keys。"
            )
        metric_values = batch.non_tensor_batch[self.metric]
        loss_masked = batch.non_tensor_batch.get(
            "loss_masked", np.zeros(len(batch), dtype=bool)
        )
        grouped_indices: dict[object, list[int]] = {}
        for index, uid in enumerate(uids):
            grouped_indices.setdefault(uid, []).append(index)

        kept_indices: list[int] = []
        for uid, indices in grouped_indices.items():
            if len(indices) != self.rollout_n:
                raise ValueError(
                    f"题组 {uid!r} 应有 {self.rollout_n} 条轨迹，实际为 {len(indices)} 条"
                )
            valid_indices = [index for index in indices if not bool(loss_masked[index])]
            values = np.asarray([metric_values[index] for index in valid_indices], dtype=np.float64)
            if len(values) >= 2 and np.std(values) > 0:
                kept_indices.extend(indices)

        generated_prompts = len(grouped_indices)
        kept_prompts = len(kept_indices) // self.rollout_n
        self._generation_batches += 1
        self._generated_prompts += generated_prompts
        self._kept_prompts += kept_prompts
        self._dropped_prompts += generated_prompts - kept_prompts
        if kept_indices:
            self._kept_batches.append(batch[kept_indices])

        cap_reached = (
            self.max_num_gen_batches > 0
            and self._generation_batches >= self.max_num_gen_batches
        )
        partial = cap_reached and 0 < self._kept_prompts < self.target_num_prompts
        stats = self._stats(max_hit=partial)
        if self._kept_prompts < self.target_num_prompts:
            if cap_reached and self._kept_prompts == 0:
                raise DynamicSamplingExhaustedError(
                    f"动态采样在 {self._generation_batches} 个生成批次后仍为 0 个可训练题组；"
                    f"共生成 {self._generated_prompts} 个题组，已全部丢弃。"
                )
            if partial:
                merged = DataProto.concat(self._kept_batches)
                merged.meta_info["dynamic_sampling_effective_prompt_batch_size"] = (
                    self._kept_prompts
                )
                result = DynamicSamplingResult(
                    ready=True,
                    batch=merged,
                    stats=stats,
                    partial=True,
                )
                self._reset()
                return result
            return DynamicSamplingResult(ready=False, batch=None, stats=stats)

        merged = DataProto.concat(self._kept_batches)
        target_trajectories = self.target_num_prompts * self.rollout_n
        result = DynamicSamplingResult(ready=True, batch=merged[:target_trajectories], stats=stats)
        self._reset()
        return result

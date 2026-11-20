"""Exclusive HTTP environment leases for multi-turn AppWorld interaction.

Each server hosts one task world at a time. A shared Ray actor coordinates
exclusive leases; verl workers access the environment through HTTP without
importing AppWorld into the worker process.

APPWORLD_PORTS_FILE specifies the server ports. APPWORLD_ENV_HOST defaults
to localhost. Execution, initialization, and evaluation timeouts are
configurable through their corresponding APPWORLD_*_TIMEOUT_S variables.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import aiohttp
import ray

logger = logging.getLogger(__file__)

POOL_ACTOR_NAME = "appworld_port_pool"
# HTTP deadlines are separate from the execution limit inside AppWorld.
EXECUTE_TIMEOUT_S = float(os.environ.get("APPWORLD_EXECUTE_TIMEOUT_S", "180"))
# Loading a task can take longer than executing one turn.
INIT_TIMEOUT_S = float(os.environ.get("APPWORLD_INIT_TIMEOUT_S", "480"))
EVAL_TIMEOUT_S = float(os.environ.get("APPWORLD_EVAL_TIMEOUT_S", "180"))


@ray.remote(num_cpus=0)
class AppWorldPortPool:
    """全局端口租约。acquire 无空闲时挂起等待;bad=True 的端口不再归还(server 疑似卡死)。"""

    def __init__(self, ports: list[int]):
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        for p in ports:
            self._queue.put_nowait(p)
        self._total = len(ports)
        self._leased: set[int] = set()
        self._bad: list[int] = []
        lease_dir = os.environ.get("APPWORLD_LEASE_DIR")
        self._lease_dir = Path(lease_dir) if lease_dir else None
        if self._lease_dir is not None:
            self._lease_dir.mkdir(parents=True, exist_ok=True)
            # 新 actor 代表新尝试；清理上次崩溃遗留的租约标记。
            for port in ports:
                (self._lease_dir / f"{port}.leased").unlink(missing_ok=True)

    def _mark_leased(self, port: int) -> None:
        if self._lease_dir is not None:
            (self._lease_dir / f"{port}.leased").write_text(f"pid={os.getpid()}\n")

    def _mark_released(self, port: int) -> None:
        if self._lease_dir is not None:
            (self._lease_dir / f"{port}.leased").unlink(missing_ok=True)

    async def acquire(self) -> int:
        # 排队是常态(并发≈池容量);烂穿风险由 _rehab 复活机制+外层 wait_for 超时兜底
        if self._queue.qsize() == 0 and len(self._bad) >= max(1, self._total // 4):
            logger.warning("端口池紧张(排队中): %s", self.status())
        port = await self._queue.get()
        self._leased.add(port)
        self._mark_leased(port)
        return port

    def release(self, port: int, bad: bool = False) -> bool:
        if port not in self._leased:
            logger.error("重复/非法 release port %s(未在租约中),拒绝入队。status=%s", port, self.status())
            return False
        self._leased.discard(port)
        self._mark_released(port)
        if bad:
            self._bad.append(port)
            logger.warning("port %s 标坏进缓刑(共 %s 坏/%s 总)", port, len(self._bad), self._total)
            asyncio.create_task(self._rehab(port))  # 外部 watchdog 会修活 server,延迟探活捞回
        else:
            self._queue.put_nowait(port)
        return True

    async def _rehab(self, port: int) -> None:
        """坏端口缓刑复活:等 watchdog 修活 server(5 分钟巡检)后探活捞回队列,防 bad 只增不减烂穿池子。"""
        for _ in range(48):  # 最多缓刑 48×420s≈5.6h,之后放弃(server 真废了)
            await asyncio.sleep(420)
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"http://localhost:{port}/docs", timeout=aiohttp.ClientTimeout(total=3)
                    ) as r:
                        if r.status == 200:
                            if port in self._bad:
                                self._bad.remove(port)
                                self._queue.put_nowait(port)
                                logger.warning("port %s 复活入队(剩 %s 坏)", port, len(self._bad))
                            return
            except Exception:  # noqa: BLE001 — 还没活,下一轮再探
                pass
        logger.error("port %s 缓刑期满未复活,永久弃用", port)

    def status(self) -> dict:
        return {
            "total": self._total,
            "free": self._queue.qsize(),
            "leased": len(self._leased),
            "bad": list(self._bad),
        }


def get_port_pool() -> "ray.actor.ActorHandle":
    """取(或建)named 租约 actor。多 worker 并发调用安全(get_if_exists)。"""
    ports_file = os.environ["APPWORLD_PORTS_FILE"]
    with open(ports_file) as f:
        ports = [int(x) for x in f.read().split() if x.strip()]
    if not ports:
        raise RuntimeError(f"端口文件为空: {ports_file}")
    # Keep leases alive across short-lived rollout handles.
    return AppWorldPortPool.options(name=POOL_ACTOR_NAME, get_if_exists=True, lifetime="detached").remote(ports)


async def acquire_port(pool: "ray.actor.ActorHandle") -> int:
    await wait_for_rollout_resume()
    timeout = float(os.environ.get("APPWORLD_ACQUIRE_TIMEOUT_S", "900"))
    return await asyncio.wait_for(pool.acquire.remote(), timeout=timeout)


async def wait_for_rollout_resume() -> None:
    """任一适用暂停标记存在时不启动新 rollout；在途租约自然完成。

    `APPWORLD_PAUSE_FILES` 用 `os.pathsep` 分隔多个标记。所有臂都订阅节点内存标记，
    仅 keep 三臂额外订阅 C4 标记。保留单数变量兼容旧 smoke。这里有意不设超时。
    """
    raw = os.environ.get("APPWORLD_PAUSE_FILES") or os.environ.get("APPWORLD_PAUSE_FILE", "")
    pause_files = [path for path in raw.split(os.pathsep) if path]
    if not pause_files:
        return
    interval = float(os.environ.get("APPWORLD_PAUSE_POLL_S", "5"))
    warned_at = 0.0
    loop = asyncio.get_running_loop()
    while active := [path for path in pause_files if os.path.exists(path)]:
        now = loop.time()
        if warned_at == 0.0 or now - warned_at >= 60:
            logger.warning("PAUSE_ROLLOUT 标记存在，acquire 前等待: %s", active)
            warned_at = now
        await asyncio.sleep(interval)


class AppWorldEnvClient:
    """单任务生命周期的 HTTP 客户端。用法:

    client = AppWorldEnvClient(port, session)
    await client.initialize(task_id, experiment_name)
    out = await client.execute(code)
    ev = await client.evaluate()          # dict(passes/failures/success...)
    done = await client.task_completed()
    await client.close()                  # 幂等
    """

    def __init__(self, port: int, session: aiohttp.ClientSession, host: Optional[str] = None):
        self.base = f"http://{host or os.environ.get('APPWORLD_ENV_HOST', 'localhost')}:{port}"
        self.port = port
        self._session = session
        self.task_id: Optional[str] = None
        self.suspect_bad = False  # execute/evaluate 超时 → server 可能被死循环代码卡死

    async def _post(self, path: str, body: dict, timeout_s: float, retries: int = 2) -> Any:
        last: Optional[Exception] = None
        for i in range(retries + 1):
            try:
                async with self._session.post(
                    f"{self.base}{path}", json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)
                ) as r:
                    if r.status != 200:
                        text = await r.text()
                        raise RuntimeError(f"{path} HTTP {r.status}: {text[:300]}")
                    return await r.json()
            except (aiohttp.ClientConnectionError, aiohttp.ClientOSError) as e:
                last = e  # 连接类错误可重试
                await asyncio.sleep(2 * (i + 1))
            except asyncio.TimeoutError as e:
                self.suspect_bad = True  # 超时不重试:server 可能已被用户代码卡死
                raise RuntimeError(f"{path} 超时 {timeout_s}s(port {self.port},server 标坏)") from e
        raise RuntimeError(f"{path} 连接失败(port {self.port}): {last}") from last

    async def initialize(self, task_id: str, experiment_name: str) -> dict:
        out = await self._post(
            "/initialize",
            {"task_id": task_id, "experiment_name": experiment_name, "raise_on_extra_parameters": False},
            INIT_TIMEOUT_S,
        )
        self.task_id = task_id
        return out

    async def execute(self, code: str) -> str:
        out = await self._post("/execute", {"task_id": self.task_id, "code": code}, EXECUTE_TIMEOUT_S, retries=0)
        return str(out["output"])

    async def evaluate(self, timeout_s: float = EVAL_TIMEOUT_S) -> dict:
        out = await self._post("/evaluate", {"task_id": self.task_id, "suppress_errors": True}, timeout_s)
        return out["output"]

    async def task_completed(self) -> bool:
        out = await self._post("/task_completed", {"task_id": self.task_id}, 60)
        return bool(out["output"])

    async def close(self) -> None:
        if self.task_id is None:
            return
        try:
            await self._post("/close", {"task_id": self.task_id}, 60, retries=0)
        except Exception as e:  # noqa: BLE001 — close 失败只降级为标坏,不炸 rollout
            logger.warning("close 失败(port %s): %s", self.port, e)
            self.suspect_bad = True
        self.task_id = None


_HTTP_SESSION: Optional[aiohttp.ClientSession] = None


def get_http_session() -> aiohttp.ClientSession:
    """每进程一个连接池(须在事件循环内首次调用)。"""
    global _HTTP_SESSION
    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        _HTTP_SESSION = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=256, limit_per_host=4))
    return _HTTP_SESSION

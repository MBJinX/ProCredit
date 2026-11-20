"""Minimal single-node ProCredit training with one local AppWorld environment."""
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from tasks import DATA, ROOT, prompt, selected_tasks


def write_dataset(path: Path, ids: list[str], split: str) -> None:
    rows = [
        {"data_source": "appworld/react", "agent_name": "appworld_react_agent",
         "prompt": prompt(task_id), "reward_model": {"style": "appworld", "ground_truth": ""},
         "extra_info": {"task_id": task_id, "split": split}}
        for task_id in ids
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


@contextmanager
def environment(out: Path):
    # Pass a bound socket to uvicorn so concurrent runs cannot lease the same port.
    with socket.socket() as sock, (out / "environment.log").open("w") as log:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        cmd = [os.environ["APPWORLD_PYTHON"], "-m", "uvicorn", "appworld.serve.environment:app",
               "--fd", str(sock.fileno())]
        server = subprocess.Popen(cmd, pass_fds=(sock.fileno(),), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"AppWorld exited. See {log.name}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/docs", timeout=2):
                        break
                except (URLError, TimeoutError):
                    time.sleep(0.5)
            else:
                raise RuntimeError(f"AppWorld startup timed out. See {log.name}")
            (out / "ports.txt").write_text(f"{port}\n")
            yield
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    model = os.environ.get("TRAIN_MODEL")
    if not model:
        raise SystemExit("Set TRAIN_MODEL to a Hugging Face model ID or local model directory")
    if Path(model).exists():
        model = str(Path(model).resolve())
    ids = selected_tasks()
    train = set((DATA / "datasets/train.txt").read_text().split())
    if set(ids) - train:
        raise SystemExit("Training requires TASK_IDS from data/datasets/train.txt")
    gpus = int(os.environ.get("TRAIN_GPUS", "1"))
    n = int(os.environ.get("ROLLOUTS_PER_TASK", "2"))
    steps = int(os.environ.get("TRAIN_STEPS", "5"))
    if gpus < 1 or n < 2 or n % gpus or steps < 1:
        raise ValueError("Use positive GPU and step counts and at least two rollouts divisible by TRAIN_GPUS")
    out = ROOT / "outputs" / ("train_" + uuid4().hex)
    out.mkdir(parents=True)
    write_dataset(out / "train.parquet", ids, "train")
    write_dataset(out / "dev.parquet", (DATA / "datasets/dev.txt").read_text().split()[:1], "val")
    worker_env = {
        "VERL_IMPORT_MODULES": "appworld_tool.aw_gigpo", "PYTHONPATH": os.environ["PYTHONPATH"],
        "APPWORLD_PORTS_FILE": str(out / "ports.txt"), "APPWORLD_ENV_HOST": "127.0.0.1",
        "APPWORLD_STEP_CREDIT": "1", "APPWORLD_STEP_TERMINAL": os.getenv("APPWORLD_STEP_TERMINAL", "keep"),
        "APPWORLD_SHAPING": "0", "APPWORLD_SHAPING_COEF": os.getenv("APPWORLD_SHAPING_COEF", "0.5"),
        "APPWORLD_EVAL_EVERY_TURN": "1", "APPWORLD_INFRA_MASK": "1", "APPWORLD_CTX_BUDGET": "8192",
        "APPWORLD_MAX_TURNS": os.getenv("APPWORLD_MAX_TURNS", "12"), "APPWORLD_GIGPO_GAMMA": "1.0",
        "APPWORLD_GIGPO_GROUP": "task", "APPWORLD_GIGPO_MODE": "mean_norm", "APPWORLD_GIGPO_W": "1.0",
        "APPWORLD_ADAPTIVE_SPREAD": "0", "APPWORLD_GROUP_GAMMA_ROUTE": "0", "APPWORLD_TOKEN_SPREAD": "0",
    }
    overrides = [f"+ray_kwargs.ray_init.runtime_env.env_vars.{k}={json.dumps(v)}" for k, v in worker_env.items()]
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides,
           "+ray_kwargs.ray_init.address=local", "trainer.use_v1=False",
           "algorithm.adv_estimator=appworld_phi_gigpo", "algorithm.use_kl_in_reward=False",
           "algorithm.rollout_correction.rollout_is=null", "algorithm.rollout_correction.bypass_mode=false",
           f'data.train_files={json.dumps(str(out / "train.parquet"))}',
           f'data.val_files={json.dumps(str(out / "dev.parquet"))}',
           "data.train_batch_size=1", "data.max_prompt_length=4096", "data.max_response_length=4096",
           "data.return_raw_chat=True", "+data.apply_chat_template_kwargs.enable_thinking=False",
           f"actor_rollout_ref.model.path={json.dumps(model)}", "actor_rollout_ref.actor.optim.lr=1e-6",
           "actor_rollout_ref.actor.ppo_mini_batch_size=1", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
           "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.rollout.name=vllm",
           "actor_rollout_ref.rollout.mode=async", f"actor_rollout_ref.rollout.n={n}",
           "actor_rollout_ref.rollout.max_model_len=8192", "actor_rollout_ref.rollout.prompt_length=8192",
           "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
           "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
           'actor_rollout_ref.rollout.agent.default_agent_loop=appworld_react_agent',
           f'actor_rollout_ref.rollout.agent.agent_loop_config_path={json.dumps(str(ROOT / "scripts/agent_loop.yaml"))}',
           f"trainer.n_gpus_per_node={gpus}", "trainer.nnodes=1", f"trainer.total_training_steps={steps}",
           f"trainer.total_epochs={steps}", "trainer.save_freq=5", "trainer.test_freq=-1",
           "trainer.val_before_train=False", "trainer.project_name=procredit", f"trainer.experiment_name={out.name}",
           'trainer.logger=["console","tensorboard"]', f'trainer.default_local_dir={json.dumps(str(out / "checkpoints"))}']
    print(f"Writing training files to {out}", flush=True)
    with environment(out), (out / "train.log").open("w") as log:
        subprocess.run(cmd, env={**os.environ, **worker_env}, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()

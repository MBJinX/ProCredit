# ProCredit

Run an AppWorld task, check the simulated world after each action, and see how much reward each turn earns. The same reward implementation feeds the included verl training recipe.

## Run your first task

Use Linux, Bash, Git and Python 3.11 or 3.12. The example runs on CPU and needs no model or API key.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
cp .env.example .env
./run.sh setup
./run.sh
```

The setup command installs a pinned AppWorld revision, unpacks its official application bundle and downloads its public task data. Use an empty directory on **node-local storage** for `APPWORLD_ROOT`. The default uses Linux shared memory and is cleared at reboot. For an existing installation, set `APPWORLD_PYTHON` and `APPWORLD_ROOT` in `.env` and go straight to `./run.sh`. [AppWorld installation and data](https://github.com/stonybrooknlp/appworld#installation)

The default example inspects the app catalog in two turns. Expect `"success": false` and a score of zero because reading the catalog leaves the task unfinished. Each run prints one JSON record per task and saves it in `outputs/<run>/results.jsonl`. The `progress` list starts with the initial state. Each following entry and each `step_rewards` entry correspond to an executed turn. AppWorld keeps its detailed execution logs under `$APPWORLD_ROOT/experiments/outputs/<run>/`.

## Choose tasks and actions

Set `TASK_IDS` in `.env` to a comma-separated list of IDs from `$APPWORLD_ROOT/data/datasets/train.txt` or `dev.txt`. Set `ACTIONS_FILE` to your own JSON file. Each string is one Python snippet executed inside the AppWorld task. Variables persist between turns, and each task starts with its own world. Relative paths are resolved from the repository root. [Example action file](examples/inspect.json)

To let a model choose actions, set these values in the same `.env` and run `./run.sh`.

```dotenv
RUN_MODE=model
MODEL_BASE_URL=http://127.0.0.1:8000/v1
MODEL_NAME=your-served-model
MODEL_API_KEY=
APPWORLD_MAX_TURNS=12
```

The endpoint must support `POST /chat/completions`. It receives the one-shot ReAct prompt, the simulated task and the tool observations. The runner executes the final Python block from each reply and stops when the task is marked complete or reaches the turn limit. Fill in a key only if your endpoint requires one. Keep your own settings in the ignored `.env`. For a second configuration, use `ENV_FILE=/path/to/another.env ./run.sh`. The launcher reads trusted Bash assignments.

## View the replay

Open `ui/index.html` directly, or run `./run.sh ui` and open `http://127.0.0.1:8080`. Change `UI_PORT` to use another port. The page contains two illustrative alarm tasks. Switch examples or pause the animation to inspect an action and the checks it changed. All assets are local. The server exposes only the `ui/` directory.

## Train with verl

Training requires a separate CUDA environment with a compatible PyTorch/vLLM stack and FlashAttention 2 matched to that stack. Allow enough GPU memory for the model, optimizer and rollouts. Install the bundled verl into that environment. Keep the quick-start environment for the AppWorld process.

```bash
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -e './verl[vllm]'
.venv-train/bin/python -m pip install --no-build-isolation 'flash-attn>=2,<3'
```

Set `APPWORLD_PYTHON` and `TRAIN_PYTHON` to the absolute paths of the two environments' Python executables. Set `TRAIN_MODEL` to a Hugging Face chat model ID or a local checkpoint, and pick training IDs from `train.txt`. Then run:

```bash
./run.sh train
```

The single-node example builds the selected prompts, starts a loopback AppWorld server, and runs the registered `appworld_phi_gigpo` estimator. Each batch uses one selected task and `ROLLOUTS_PER_TASK` independent attempts (default 2). Choose at least two attempts and a count divisible by `TRAIN_GPUS`. The default run covers five updates with a checkpoint every five steps. Logs and checkpoints go under `outputs/train_<id>/`. This is a small training configuration with an 8192-token context and one environment process. The paper used larger batches and longer runs.

## Implementation

| File | Role |
| --- | --- |
| [Task runner](scripts/run_tasks.py) | Executes actions or model replies and records per-turn scores |
| [Task preparation](scripts/tasks.py) | Builds the one-shot prompt with the installed task's simulated supervisor |
| [AppWorld client](verl/recipe/appworld_tool/aw_client.py) | Leases a world to one training rollout at a time |
| [Agent loop](verl/recipe/appworld_tool/aw_agent_loop.py) | Tracks turns and attaches each reward to its final model token |
| [Reward](verl/recipe/appworld_tool/aw_reward.py) | Computes verified progress and turn rewards |
| [Advantage estimator](verl/recipe/appworld_tool/aw_gigpo.py) | Combines task-level and turn-level credit |

Progress is the fraction of checks labelled `no_op_fail` that currently pass. The default reward is half the change in progress at each turn, plus the binary outcome on the final turn. The training entry uses task grouping, discount 1, mean-centered turn returns, and weight 1 for the turn-level advantage. It disables adaptive, routed-discount and token-spread ablations.

The release code uses Apache-2.0. The bundled verl tree retains its upstream license and notices. Downloaded AppWorld code and data are subject to their upstream redistribution terms. [License](LICENSE) · [verl notices](verl/Notice.txt) · [AppWorld license](https://github.com/stonybrooknlp/appworld#license) · [GiGPO](https://arxiv.org/abs/2505.10978)

"""Execute user-selected tasks and record only numeric progress and rewards."""
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import requests
from appworld import AppWorld
from appworld_tool import aw_reward
from tasks import ROOT, prompt, selected_tasks


def next_model_action(messages: list[dict]) -> str:
    url = os.environ["MODEL_BASE_URL"].rstrip("/") + "/chat/completions"
    token = os.environ.get("MODEL_API_KEY")
    response = requests.post(
        url, headers={"Authorization": f"Bearer {token}"} if token else {},
        json={"model": os.environ["MODEL_NAME"], "messages": messages,
              "temperature": 1.0, "top_p": 1.0, "max_tokens": 512},
        timeout=180, allow_redirects=False,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    if not isinstance(text, str):
        raise ValueError("The endpoint must return message.content as text")
    messages.append({"role": "assistant", "content": text})
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text[-100:]


def main():
    task_ids = selected_tasks()
    mode = os.environ.get("RUN_MODE", "actions")
    limit = int(os.environ.get("APPWORLD_MAX_TURNS", "12"))
    if limit < 1:
        raise ValueError("APPWORLD_MAX_TURNS must be positive")
    actions = []
    if mode == "actions":
        actions = json.loads(Path(os.environ["ACTIONS_FILE"]).read_text())
        if not isinstance(actions, list) or not actions or not all(isinstance(a, str) and a.strip() for a in actions):
            raise ValueError("ACTIONS_FILE must contain a nonempty JSON array of Python strings")
    elif mode != "model" or not all(os.environ.get(k) for k in ("MODEL_BASE_URL", "MODEL_NAME")):
        raise ValueError("Choose RUN_MODE=actions or configure MODEL_BASE_URL and MODEL_NAME for model mode")

    run_id = "procredit_" + uuid4().hex
    out = ROOT / "outputs" / run_id
    out.mkdir(parents=True)
    print(f"Writing results to {out}/results.jsonl", flush=True)
    os.environ.setdefault("APPWORLD_STEP_TERMINAL", "keep")
    with (out / "results.jsonl").open("x") as stream:
        for task_id in task_ids:
            messages = prompt(task_id) if mode == "model" else []
            with AppWorld(task_id=task_id, experiment_name=run_id, raise_on_extra_parameters=False) as world:
                evaluation = world.evaluate().to_dict()
                phi = [aw_reward.phi_of(evaluation)]
                for i in range(min(limit, len(actions)) if mode == "actions" else limit):
                    action = actions[i] if mode == "actions" else next_model_action(messages)
                    observation = world.execute(action)
                    if mode == "model":
                        messages.append({"role": "user", "content": f"Output:\n```\n{observation}\n```"})
                    evaluation = world.evaluate().to_dict()
                    phi.append(aw_reward.phi_of(evaluation))
                    if world.task_completed():
                        break
                success = evaluation["success"]
                rewards = aw_reward.compute_step_rewards(float(success), phi, aw_reward.shaping_coef())
                result = {"task_id": task_id, "success": success, "turns": len(rewards),
                          "progress": phi, "step_rewards": rewards, "score": sum(rewards)}
                line = json.dumps(result, allow_nan=False)
                print(line, file=stream, flush=True)
                print(line, flush=True)


if __name__ == "__main__":
    main()

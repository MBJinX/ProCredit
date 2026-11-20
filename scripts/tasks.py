"""Build AppWorld tasks and the one-shot ReAct prompt from local data."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["APPWORLD_ROOT"]) / "data"


def selected_tasks() -> list[str]:
    ids = [part.strip() for part in os.environ.get("TASK_IDS", "").split(",")]
    for task_id in ids:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError("TASK_IDS must be a comma-separated list of installed task IDs")
        if not (DATA / "tasks" / task_id / "specs.json").is_file():
            raise ValueError(f"Task {task_id} is missing from APPWORLD_ROOT/data/tasks")
    if len(set(ids)) != len(ids):
        raise ValueError("TASK_IDS contains duplicate tasks")
    return ids


def prompt(task_id: str) -> list[dict]:
    spec = json.loads((DATA / "tasks" / task_id / "specs.json").read_text())
    person = spec["supervisor"]
    name = f"{person['first_name']} {person['last_name']}"
    email = person["email"]
    phone = person["phone_number"]
    prefix = (ROOT / "prompts/react_prefix.txt").read_text()
    prefix = prefix.replace("{{name}}", name).replace("{{email}}", email).replace("{{phone}}", phone)
    suffix = (ROOT / "prompts/react_suffix.txt").read_text()
    task = f"\n\nMy name is: {name}. My personal email is {email} and phone number is {phone}.\nTask: {spec['instruction']}\n\n"
    return [{"role": "user", "content": prefix + task + suffix + "\n"}]

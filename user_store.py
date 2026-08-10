import json
import os
from datetime import datetime

STORE_PATH = "users.json"


def _load():
    if not os.path.exists(STORE_PATH):
        return {"users": {}}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def user_exists(email: str) -> bool:
    data = _load()
    return email.lower() in data["users"]


def create_user(email: str, password_hash: str):
    data = _load()
    data["users"][email.lower()] = {
        "password_hash": password_hash,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save(data)


def get_user(email: str):
    data = _load()
    return data["users"].get(email.lower())
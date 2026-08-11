import json
import os
import uuid
from datetime import datetime

STORE_PATH = "chat_history.json"


def _load():
    if not os.path.exists(STORE_PATH):
        return {"sessions": {}}
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_session(user_email):
    data = _load()
    session_id = str(uuid.uuid4())
    data["sessions"][session_id] = {
        "user": user_email,
        "title": "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
    }
    _save(data)
    return session_id


def add_message(session_id, role, content, user_email):
    data = _load()
    if session_id not in data["sessions"]:
        data["sessions"][session_id] = {
            "user": user_email,
            "title": "New Chat",
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
        }

    session = data["sessions"][session_id]
    session["messages"].append({"role": role, "content": content})

    if role == "user" and session["title"] == "New Chat":
        session["title"] = content[:40] + ("..." if len(content) > 40 else "")

    _save(data)


def get_sessions(user_email):
    data = _load()
    sessions = [
        {"id": sid, "title": s["title"], "created_at": s["created_at"]}
        for sid, s in data["sessions"].items()
        if s.get("user") == user_email
    ]
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return sessions


def get_messages(session_id, user_email):
    data = _load()
    session = data["sessions"].get(session_id)
    if not session or session.get("user") != user_email:
        return []  # not found, or belongs to someone else
    return session["messages"]


def delete_session(session_id, user_email):
    data = _load()
    session = data["sessions"].get(session_id)
    if not session or session.get("user") != user_email:
        return False  # not found, or not yours to delete
    del data["sessions"][session_id]
    _save(data)
    return True
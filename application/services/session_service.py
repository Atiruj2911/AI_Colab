"""SessionService: persist session metadata, track active run, and support UI reconnect."""

from __future__ import annotations

import uuid
from typing import Any

from core.schemas import utc_now_iso
from infrastructure.process.state_store import read_json, write_json_atomic
from infrastructure.storage.workspace_manager import WorkspaceManager


class SessionService:
    """Persists current studio session metadata inside workspace/sessions/."""

    def __init__(self, workspace: WorkspaceManager) -> None:
        self.workspace = workspace
        self.current_session_file = self.workspace.sessions_dir / "current_session.json"

    def get_or_create_session(self) -> dict[str, Any]:
        existing = read_json(self.current_session_file)
        if existing and existing.get("session_id"):
            return existing

        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        now = utc_now_iso()
        session = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "active_dataset_id": None,
            "active_run_id": None,
            "last_config": {},
        }
        write_json_atomic(self.current_session_file, session)
        return session

    def update_session(self, **updates: Any) -> dict[str, Any]:
        session = self.get_or_create_session()
        session.update(updates)
        session["updated_at"] = utc_now_iso()
        write_json_atomic(self.current_session_file, session)
        return session

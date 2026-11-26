"""Unit tests for background job lifecycle: duplicate prevention, dead-PID reconciliation, stop & reconnect."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from core.schemas import JobState, TrainRequest
from infrastructure.process.job_manager import JobManager
from infrastructure.process.state_store import write_json_atomic
from infrastructure.storage.workspace_manager import WorkspaceManager


class TestJobLifecycle(unittest.TestCase):
    def test_dead_pid_is_reconciled_to_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = WorkspaceManager(Path(tmp) / "ws")
            jm = JobManager(ws, Path(__file__).resolve().parent.parent)

            run_dir = ws.runs_dir / "dead_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            state = JobState(
                job_id="dead_run",
                session_id="sess_1",
                run_id="dead_run",
                run_name="dead_run",
                dataset_id="ds_1",
                model_family="yolo11",
                task="detect",
                model_variant="yolo11n",
                status="running",
                pid=9999999,  # non-existent PID
                run_dir=str(run_dir),
            )
            write_json_atomic(run_dir / "job_state.json", state.to_dict())
            write_json_atomic(jm.active_job_pointer, {"run_id": "dead_run"})

            reconciled = jm.get_active_job()
            self.assertIsNotNone(reconciled)
            self.assertEqual(reconciled.status, "failed")
            self.assertIsNotNone(reconciled.error_message)

    def test_duplicate_start_blocked_when_active_pid_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = WorkspaceManager(Path(tmp) / "ws")
            jm = JobManager(ws, Path(__file__).resolve().parent.parent)

            run_dir = ws.runs_dir / "live_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            # Use current process PID to simulate an alive background worker
            live_state = JobState(
                job_id="live_run",
                session_id="sess_1",
                run_id="live_run",
                run_name="live_run",
                dataset_id="ds_1",
                model_family="yolo11",
                task="detect",
                model_variant="yolo11n",
                status="running",
                pid=os.getpid(),
                run_dir=str(run_dir),
            )
            write_json_atomic(run_dir / "job_state.json", live_state.to_dict())
            write_json_atomic(jm.active_job_pointer, {"run_id": "live_run"})

            req = TrainRequest(
                session_id="sess_1",
                run_id="second_run",
                run_name="second_run",
                dataset_id="ds_1",
                dataset_yaml=str(Path(tmp) / "data.yaml"),
                model_variant="yolo11n",
                epochs=1,
                batch_size=2,
                imgsz=320,
            )
            with self.assertRaises(RuntimeError):
                jm.start_job(req)

            # Requesting stop writes stop_requested.flag and marks status='stopping'
            stopped = jm.request_stop("live_run", force=False)
            self.assertIsNotNone(stopped)
            self.assertTrue(stopped.stop_requested)
            self.assertEqual(stopped.status, "stopping")
            self.assertTrue((run_dir / "stop_requested.flag").exists())


if __name__ == "__main__":
    unittest.main()

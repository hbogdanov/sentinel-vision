from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.api.models import AlertEvent


class SQLiteAlertStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def ingest(self, event: AlertEvent | dict[str, Any]) -> None:
        alert = (
            event if isinstance(event, AlertEvent) else AlertEvent.model_validate(event)
        )
        payload_dict = alert.model_dump(mode="json", by_alias=True)
        payload = json.dumps(payload_dict, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO alerts (
                    event_id, timestamp, camera_id, event_type, zone,
                    track_id, class_name, confidence, frame_index,
                    snapshot_path, clip_path, metadata_path, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    camera_id=excluded.camera_id,
                    event_type=excluded.event_type,
                    zone=excluded.zone,
                    track_id=excluded.track_id,
                    class_name=excluded.class_name,
                    confidence=excluded.confidence,
                    frame_index=excluded.frame_index,
                    snapshot_path=excluded.snapshot_path,
                    clip_path=excluded.clip_path,
                    metadata_path=excluded.metadata_path,
                    payload=excluded.payload
                """,
                (
                    alert.event_id,
                    payload_dict["timestamp"],
                    alert.camera_id,
                    alert.event_type,
                    alert.zone,
                    alert.track_id,
                    alert.object_class,
                    alert.confidence,
                    alert.frame_index,
                    alert.snapshot_path or "",
                    alert.clip_path or "",
                    alert.metadata_path or "",
                    payload,
                ),
            )

    def list_alerts(
        self,
        *,
        camera_id: str | None = None,
        event_type: str | None = None,
        zone: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = "SELECT payload FROM alerts"
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if zone:
            clauses.append("zone = ?")
            params.append(zone)
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            AlertEvent.model_validate(json.loads(row["payload"])).model_dump(
                mode="json", by_alias=True
            )
            for row in rows
        ]

    def get_alert(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM alerts WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        return AlertEvent.model_validate(json.loads(row["payload"])).model_dump(
            mode="json", by_alias=True
        )

    def stats(
        self,
        *,
        camera_id: str | None = None,
        event_type: str | None = None,
        zone: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        where_sql, params = self._filters_to_sql(
            camera_id=camera_id,
            event_type=event_type,
            zone=zone,
            start_time=start_time,
            end_time=end_time,
        )
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) AS count FROM alerts {where_sql}", params
            ).fetchone()["count"]
            by_event_type_rows = connection.execute(
                f"SELECT event_type, COUNT(*) AS count FROM alerts {where_sql} GROUP BY event_type ORDER BY count DESC",
                params,
            ).fetchall()
            by_camera_rows = connection.execute(
                f"SELECT camera_id, COUNT(*) AS count FROM alerts {where_sql} GROUP BY camera_id ORDER BY count DESC",
                params,
            ).fetchall()
            latest = connection.execute(
                f"SELECT timestamp FROM alerts {where_sql} ORDER BY timestamp DESC LIMIT 1",
                params,
            ).fetchone()
        return {
            "total_alerts": int(total),
            "by_event_type": {
                row["event_type"]: int(row["count"]) for row in by_event_type_rows
            },
            "by_camera_id": {
                row["camera_id"]: int(row["count"]) for row in by_camera_rows
            },
            "latest_timestamp": latest["timestamp"] if latest else None,
            "db_path": str(self.db_path).replace("\\", "/"),
        }

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM alerts").fetchone()
        return int(row["count"])

    def distinct_values(self, column: str) -> list[str]:
        if column not in {"camera_id", "event_type", "zone"}:
            raise ValueError(f"Unsupported distinct column: {column}")
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT DISTINCT {column} AS value FROM alerts WHERE {column} != '' ORDER BY {column} ASC"
            ).fetchall()
        return [str(row["value"]) for row in rows]

    def camera_summaries(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT camera_id, COUNT(*) AS alerts, MAX(timestamp) AS latest_timestamp
                FROM alerts
                WHERE camera_id != ''
                GROUP BY camera_id
                ORDER BY camera_id ASC
                """).fetchall()
        return [
            {
                "camera_id": str(row["camera_id"]),
                "alerts": int(row["alerts"]),
                "latest_timestamp": row["latest_timestamp"],
            }
            for row in rows
        ]

    def _filters_to_sql(
        self,
        *,
        camera_id: str | None,
        event_type: str | None,
        zone: str | None,
        start_time: str | None,
        end_time: str | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if zone:
            clauses.append("zone = ?")
            params.append(zone)
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT,
                    event_type TEXT,
                    zone TEXT,
                    track_id INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    frame_index INTEGER,
                    snapshot_path TEXT,
                    clip_path TEXT,
                    metadata_path TEXT,
                    payload TEXT NOT NULL
                )
                """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_camera_id ON alerts(camera_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_event_type ON alerts(event_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_zone ON alerts(zone)"
            )

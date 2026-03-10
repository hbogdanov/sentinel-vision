from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from src.api.storage import SQLiteAlertStore


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit dashboard dependencies are not installed. Install `streamlit` and `pandas`."
        ) from exc

    store = SQLiteAlertStore(
        os.getenv("SENTINEL_ALERTS_DB_PATH", "data/outputs/alerts.db")
    )

    st.set_page_config(page_title="Sentinel Vision Dashboard", layout="wide")
    st.title("Sentinel Vision Alerts")

    with st.sidebar:
        st.header("Filters")
        camera_options = ["All"] + store.distinct_values("camera_id")
        event_options = ["All"] + store.distinct_values("event_type")
        zone_options = ["All"] + store.distinct_values("zone")
        camera_id = st.selectbox("Camera", camera_options)
        event_type = st.selectbox("Event Type", event_options)
        zone = st.selectbox("Zone", zone_options)
        default_start = datetime.now(timezone.utc) - timedelta(days=7)
        start_time = st.text_input(
            "Start Time (UTC ISO-8601)",
            default_start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        end_time = st.text_input(
            "End Time (UTC ISO-8601)",
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        limit = st.slider("Rows", min_value=10, max_value=500, value=100, step=10)

    filters = {
        "camera_id": None if camera_id == "All" else camera_id,
        "event_type": None if event_type == "All" else event_type,
        "zone": None if zone == "All" else zone,
        "start_time": start_time or None,
        "end_time": end_time or None,
    }
    stats = store.stats(**filters)
    alerts = store.list_alerts(limit=limit, offset=0, **filters)
    frame = pd.DataFrame(alerts)

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Total Alerts", stats["total_alerts"])
    metric_col_2.metric("Latest Alert", stats["latest_timestamp"] or "None")
    metric_col_3.metric("Database", stats["db_path"])

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.subheader("By Event Type")
        if stats["by_event_type"]:
            event_df = pd.DataFrame(
                [
                    {"event_type": key, "count": value}
                    for key, value in stats["by_event_type"].items()
                ]
            ).set_index("event_type")
            st.bar_chart(event_df)
        else:
            st.info("No alerts in the selected window.")

    with chart_col_2:
        st.subheader("By Camera")
        if stats["by_camera_id"]:
            camera_df = pd.DataFrame(
                [
                    {"camera_id": key, "count": value}
                    for key, value in stats["by_camera_id"].items()
                ]
            ).set_index("camera_id")
            st.bar_chart(camera_df)
        else:
            st.info("No camera data for the selected window.")

    st.subheader("Alerts")
    if frame.empty:
        st.info("No alerts match the current filters.")
        return

    preferred_columns = [
        "timestamp",
        "camera_id",
        "event_type",
        "zone",
        "track_id",
        "class",
        "confidence",
        "snapshot_path",
        "clip_path",
        "metadata_path",
        "event_id",
    ]
    visible_columns = [
        column for column in preferred_columns if column in frame.columns
    ]
    st.dataframe(frame[visible_columns], use_container_width=True)

    selected_event_id = st.selectbox("Alert Details", frame["event_id"].tolist())
    selected = next(alert for alert in alerts if alert["event_id"] == selected_event_id)
    st.json(selected)


if __name__ == "__main__":
    main()

# Evaluation Bundle

This directory holds the benchmark manifest, per-video labels, model outputs, and generated results for short evaluation clips.

## Layout

- `benchmark_manifest.json`: list of evaluation videos and the JSON files used for scoring.
- `annotations/`: manually maintained ground-truth detections and expected events.
- `predictions/`: detector, tracker, and event outputs exported from the system under test.
- `results/`: generated evaluation summaries.
- `videos/`: reference clips for the bundled micro-benchmark; replace or extend these with manually labeled footage as the benchmark grows.

## JSON schema

Each annotation or prediction file uses:

```json
{
  "video_id": "office_intrusion_short",
  "fps": 10,
  "duration_seconds": 6.0,
  "detections": [
    {
      "frame_index": 0,
      "track_id": 1,
      "class": "person",
      "bbox": [0, 0, 10, 20],
      "score": 0.96
    }
  ],
  "events": [
    {
      "event_type": "intrusion",
      "zone": "restricted_lab",
      "class": "person",
      "track_id": 1,
      "entry_frame_index": 2,
      "frame_index": 2,
      "match_tolerance_frames": 3
    }
  ]
}
```

## Labeling guidance

- Use 3 to 5 short clips per benchmark pass.
- Label the target classes you care about for detection and tracking.
- For intrusion and loitering, mark `entry_frame_index` and the expected `frame_index` for the alert.
- Keep `match_tolerance_frames` tight enough to catch late or early alerts.

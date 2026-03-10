from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float,
        classes_of_interest: Iterable[str] | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for detection. Install dependencies first."
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.classes_of_interest = set(classes_of_interest or [])
        self.names = self.model.names

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                label = str(self.names[class_id])
                if self.classes_of_interest and label not in self.classes_of_interest:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf.item())
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        score=score,
                        class_id=class_id,
                        label=label,
                    )
                )
        return detections

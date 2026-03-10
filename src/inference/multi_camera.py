from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.inference.pipeline import SentinelPipeline

LOGGER = logging.getLogger(__name__)


class MultiCameraRunner:
    def __init__(self, camera_configs: list[dict[str, Any]]) -> None:
        self.camera_configs = camera_configs

    def run(self) -> None:
        if len(self.camera_configs) <= 1:
            SentinelPipeline(self.camera_configs[0]).run()
            return

        with ThreadPoolExecutor(
            max_workers=len(self.camera_configs), thread_name_prefix="camera"
        ) as executor:
            futures = []
            for camera_config in self.camera_configs:
                LOGGER.info(
                    "Starting camera pipeline %s for source %s",
                    camera_config["camera_id"],
                    camera_config["input"]["source"],
                )
                futures.append(executor.submit(SentinelPipeline(camera_config).run))
            for future in futures:
                future.result()

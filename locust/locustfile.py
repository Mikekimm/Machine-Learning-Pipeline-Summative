from __future__ import annotations

from io import BytesIO
from pathlib import Path

from locust import HttpUser, between, task
from PIL import Image


def make_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(100, 140, 190))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


class ModelUser(HttpUser):
    wait_time = between(1, 2)

    @task(1)
    def check_health(self):
        self.client.get("/health")

    @task(5)
    def predict(self):
        payload = make_dummy_image_bytes()
        files = {"file": ("dummy.jpg", payload, "image/jpeg")}
        self.client.post("/predict", files=files)

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")

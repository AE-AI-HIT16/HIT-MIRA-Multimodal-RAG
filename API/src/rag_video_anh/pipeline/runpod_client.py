"""Minimal client for queue-based Runpod Serverless endpoints."""

from __future__ import annotations

import os
from typing import Any

import httpx


class RunpodClient:
    """Submit and inspect async jobs without exposing the API key to clients."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint_id: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID", "")
        self.base_url = (base_url or os.getenv("RUNPOD_API_BASE_URL", "https://api.runpod.ai/v2")).rstrip("/")
        self.client = client or httpx.Client(timeout=30.0)
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is not configured")
        if not self.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID is not configured")

    @property
    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(f"{self.base_url}/{self.endpoint_id}/run", headers=self._headers, json={"input": payload})
        response.raise_for_status()
        return response.json()

    def status(self, runpod_job_id: str) -> dict[str, Any]:
        response = self.client.get(f"{self.base_url}/{self.endpoint_id}/status/{runpod_job_id}", headers=self._headers)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        """Return endpoint worker/queue health."""

        response = self.client.get(f"{self.base_url}/{self.endpoint_id}/health", headers=self._headers)
        response.raise_for_status()
        return response.json()

    def cancel(self, runpod_job_id: str) -> dict[str, Any]:
        """Cancel a queued or in-progress job that is no longer usable."""

        response = self.client.post(f"{self.base_url}/{self.endpoint_id}/cancel/{runpod_job_id}", headers=self._headers)
        response.raise_for_status()
        return response.json()

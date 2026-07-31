"""Thin REST helpers for the local XQdrant fork (incl. with_dims_explained)."""
from __future__ import annotations
import time
from typing import Any
import requests

class XQdrantREST:

    def __init__(self, base_url: str, *, timeout: float=120.0) -> None:
        self.base = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def wait_ready(self, *, retries: int=60, sleep_s: float=0.5) -> None:
        last: Exception | None = None
        for _ in range(retries):
            try:
                r = self.session.get(f'{self.base}/readyz', timeout=2)
                if r.status_code == 200:
                    return
            except requests.RequestException as e:
                last = e
            time.sleep(sleep_s)
        raise RuntimeError(f'XQdrant not ready at {self.base}: {last}')

    def collection_exists(self, name: str) -> bool:
        r = self.session.get(f'{self.base}/collections/{name}', timeout=self.timeout)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    def delete_collection(self, name: str) -> None:
        r = self.session.delete(f'{self.base}/collections/{name}', timeout=self.timeout)
        if r.status_code not in (200, 404):
            r.raise_for_status()

    def create_collection(self, name: str, *, size: int, distance: str='Cosine', on_disk: bool=False) -> None:
        body: dict[str, Any] = {'vectors': {'size': size, 'distance': distance}}
        if on_disk:
            body['vectors']['on_disk'] = True
        r = self.session.put(f'{self.base}/collections/{name}', json=body, timeout=self.timeout)
        r.raise_for_status()

    def count(self, name: str, *, exact: bool=True) -> int:
        r = self.session.post(f'{self.base}/collections/{name}/points/count', json={'exact': exact}, timeout=self.timeout)
        r.raise_for_status()
        return int(r.json()['result']['count'])

    def upsert(self, name: str, points: list[dict[str, Any]], *, wait: bool=True) -> None:
        r = self.session.put(f'{self.base}/collections/{name}/points', params={'wait': 'true' if wait else 'false'}, json={'points': points}, timeout=max(self.timeout, 300.0))
        r.raise_for_status()

    def query(self, name: str, vector: list[float], *, limit: int=10, with_payload: bool=True, with_dims_explained: bool | dict | None=None, query_filter: dict | None=None) -> list[dict[str, Any]]:
        body: dict[str, Any] = {'query': {'nearest': vector}, 'limit': limit, 'with_payload': with_payload}
        if query_filter is not None:
            body['filter'] = query_filter
        if with_dims_explained is not None:
            body['with_dims_explained'] = with_dims_explained
        r = self.session.post(f'{self.base}/collections/{name}/points/query', json=body, timeout=self.timeout)
        r.raise_for_status()
        return list(r.json()['result']['points'])

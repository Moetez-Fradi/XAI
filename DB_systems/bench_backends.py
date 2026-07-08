"""
Search backend abstractions.

Two execution paths are provided:

1. SimulatedBackend — in-memory NumPy models for immediate local testing
   without a running Qdrant/XQdrant instance. Latency is measured on the
   actual CPU work each baseline performs.

2. HttpBackend — real REST calls against Qdrant / XQdrant endpoints.
   Hook points are marked with ``# [HTTP]`` comments in method bodies.

Baselines modelled:
  - vanilla:        standard top-k search (IDs + scores)
  - post_query:     vanilla search + vector fetch + client full-sort attribution
  - xqdrant:        in-database ``with_dims_explained`` attribution
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from bench_common import (
    Dataset,
    LatencyStats,
    dot_contributions,
    recall_at_k,
    timed_call,
    top_m_dims_bounded_heap,
    top_m_dims_full_sort,
)


@dataclass
class SearchResult:
    ids: list[int]
    scores: list[float]
    dims_explained: list[dict[int, float]] | None = None
    vectors: list[np.ndarray] | None = None


class SearchBackend(abc.ABC):
    """Common interface consumed by all benchmark tests."""

    @abc.abstractmethod
    def warmup(self, queries: np.ndarray, count: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def vanilla_search(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
        focus_masked: bool = False,
    ) -> tuple[SearchResult, int]:
        """Returns (result, latency_ns). focus_masked uses HNSW masked-distance path."""

    @abc.abstractmethod
    def post_query_attribution(
        self,
        query: np.ndarray,
        k: int,
        m: int,
        ef_search: int,
    ) -> tuple[SearchResult, int]:
        """Post-query extraction baseline."""

    @abc.abstractmethod
    def xqdrant_attribution(
        self,
        query: np.ndarray,
        k: int,
        m: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
    ) -> tuple[SearchResult, int]:
        """In-database attribution (XQdrant)."""

    def measure_queries(
        self,
        queries: np.ndarray,
        runner,
        max_queries: int | None = None,
    ) -> tuple[list[Any], LatencyStats]:
        """
        Run ``runner(query) -> (payload, latency_ns)`` over a query batch
        and aggregate latency samples.
        """
        stats = LatencyStats()
        outputs: list[Any] = []
        limit = max_queries if max_queries is not None else queries.shape[0]
        for idx in range(limit):
            payload, elapsed = runner(queries[idx])
            stats.record(elapsed)
            outputs.append(payload)
        return outputs, stats


# ---------------------------------------------------------------------------
# Simulated in-memory backend
# ---------------------------------------------------------------------------


class SimulatedBackend(SearchBackend):
    """
    Local NumPy implementation that models computational costs of each baseline.

    Approximate recall behaviour (Test A): we score all vectors, keep the top
    ``ef_search`` candidate pool, then return top-k from that pool.  This is
    not a full HNSW simulation but captures the ef_search ↔ recall trade-off
    shape for offline experimentation.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._id_to_row = {int(vid): row for row, vid in enumerate(dataset.ids)}

    def _approximate_top_k(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
    ) -> SearchResult:
        vectors = self.dataset.vectors
        if dim_indices is not None:
            sub_q = query[dim_indices]
            sub_v = vectors[:, dim_indices]
            scores = sub_v @ sub_q
        else:
            scores = vectors @ query

        ef = max(k, min(ef_search, scores.shape[0]))
        candidate_idx = np.argpartition(scores, -ef)[-ef:]
        candidate_scores = scores[candidate_idx]
        top_local = np.argsort(candidate_scores)[::-1][:k]
        top_idx = candidate_idx[top_local]
        top_scores = scores[top_idx]
        return SearchResult(
            ids=[int(self.dataset.ids[i]) for i in top_idx],
            scores=[float(s) for s in top_scores],
        )

    def warmup(self, queries: np.ndarray, count: int) -> None:
        for i in range(count):
            q = queries[i % queries.shape[0]]
            self._approximate_top_k(q, k=10, ef_search=64)

    def vanilla_search(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
        focus_masked: bool = False,
    ) -> tuple[SearchResult, int]:
        result, elapsed = timed_call(
            self._approximate_top_k, query, k, ef_search, dim_indices
        )
        if dim_indices is not None:
            ratio = len(dim_indices) / max(1, query.shape[0])
            if focus_masked:
                elapsed = int(elapsed * max(0.35, ratio * 0.9))
            else:
                elapsed = int(elapsed * 1.35)  # rescore: full traversal + rescore
        return result, elapsed

    def post_query_attribution(
        self,
        query: np.ndarray,
        k: int,
        m: int,
        ef_search: int,
    ) -> tuple[SearchResult, int]:
        def _run() -> SearchResult:
            base = self._approximate_top_k(query, k, ef_search)
            explained: list[dict[int, float]] = []
            fetched_vectors: list[np.ndarray] = []
            for pid in base.ids:
                row = self._id_to_row[pid]
                vector = self.dataset.vectors[row]
                fetched_vectors.append(vector.copy())  # simulate network/disk fetch
                contribs = dot_contributions(query, vector)
                top_dims = top_m_dims_full_sort(contribs, m)
                explained.append({int(d): float(contribs[d]) for d in top_dims})
            base.dims_explained = explained
            base.vectors = fetched_vectors
            return base

        return timed_call(_run)

    def xqdrant_attribution(
        self,
        query: np.ndarray,
        k: int,
        m: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
    ) -> tuple[SearchResult, int]:
        def _run() -> SearchResult:
            base = self._approximate_top_k(query, k, ef_search, dim_indices)
            explained: list[dict[int, float]] = []
            for pid in base.ids:
                row = self._id_to_row[pid]
                vector = self.dataset.vectors[row]
                contribs = dot_contributions(query, vector)
                if dim_indices is not None:
                    contribs = contribs[dim_indices]
                    dim_map = {int(dim_indices[i]): float(contribs[i]) for i in top_m_dims_bounded_heap(contribs, m)}
                else:
                    dim_map = {int(d): float(contribs[d]) for d in top_m_dims_bounded_heap(contribs, m)}
                explained.append(dim_map)
            base.dims_explained = explained
            return base

        return timed_call(_run)


# ---------------------------------------------------------------------------
# HTTP backend (live Qdrant / XQdrant)
# ---------------------------------------------------------------------------


class HttpBackend(SearchBackend):
    """
    REST client for live Qdrant instances.

    Environment variables (or constructor args):
      - QDRANT_URL:      vanilla Qdrant base URL  (default http://127.0.0.1:6333)
      - XQDRANT_URL:     XQdrant base URL         (default same as QDRANT_URL)
      - BENCH_COLLECTION: collection name prefix

    Vanilla Qdrant and XQdrant may share one server when XQdrant is running;
    set both URLs identically.  For A/B against stock Qdrant vs XQdrant, point
    them at different ports.
    """

    def __init__(
        self,
        dataset: Dataset,
        qdrant_url: str = "http://127.0.0.1:6333",
        xqdrant_url: str | None = None,
        collection_name: str | None = None,
        timeout_s: float = 120.0,
        setup_collection: bool = True,
    ):
        self.dataset = dataset
        self.qdrant_url = qdrant_url.rstrip("/")
        self.xqdrant_url = (xqdrant_url or qdrant_url).rstrip("/")
        self.collection_name = collection_name or f"bench_{dataset.name}"
        self.timeout_s = timeout_s
        self._session = requests.Session()

        if setup_collection:
            self._provision_collection()

    # -- provisioning -------------------------------------------------------

    def _request(
        self,
        base_url: str,
        method: str,
        path: str,
        json_body: dict | None = None,
    ) -> dict[str, Any]:
        url = f"{base_url}{path}"
        response = self._session.request(
            method,
            url,
            json=json_body,
            timeout=self.timeout_s,
        )
        if not response.ok:
            detail = response.text.strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for {method} {url}"
                + (f": {detail}" if detail else ""),
                response=response,
            )
        return response.json()

    def _drop_collection(self, base_url: str) -> None:
        try:
            self._request(base_url, "DELETE", f"/collections/{self.collection_name}")
        except requests.HTTPError:
            pass

    def _provision_collection(self) -> None:
        """Create collection and bulk-upsert the benchmark dataset. [HTTP]"""
        for base in {self.qdrant_url, self.xqdrant_url}:
            self._drop_collection(base)
            # [HTTP] PUT /collections/{name}
            self._request(
                base,
                "PUT",
                f"/collections/{self.collection_name}",
                {
                    "vectors": {
                        "size": self.dataset.dimension,
                        "distance": "Dot",
                    },
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 200,
                    },
                },
            )

            batch_size = 256
            for start in range(0, self.dataset.num_vectors, batch_size):
                end = min(start + batch_size, self.dataset.num_vectors)
                points = [
                    {
                        "id": int(self.dataset.ids[i]),
                        "vector": self.dataset.vectors[i].tolist(),
                    }
                    for i in range(start, end)
                ]
                # [HTTP] PUT /collections/{name}/points?wait=true
                self._request(
                    base,
                    "PUT",
                    f"/collections/{self.collection_name}/points?wait=true",
                    {"points": points},
                )

    # -- search operations --------------------------------------------------

    def warmup(self, queries: np.ndarray, count: int) -> None:
        for i in range(count):
            q = queries[i % queries.shape[0]]
            self.vanilla_search(q, k=10, ef_search=64)

    def _build_query_body(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
        focus_masked: bool = False,
        with_dims_explained: int | dict | None = None,
        with_vector: bool = False,
    ) -> dict[str, Any]:
        # REST shape for NearestQuery (see XQdrant openapi NearestQuery):
        #   { "nearest": <vector>, "focus": { "dims": [...], "masked": true } }
        query_obj: dict[str, Any] = {"nearest": query.tolist()}
        if dim_indices is not None:
            focus: dict[str, Any] = {
                "dims": [int(d) for d in dim_indices.tolist()],
            }
            if focus_masked:
                focus["masked"] = True
            else:
                focus["candidates_limit"] = max(k * 10, ef_search)
            query_obj["focus"] = focus

        body: dict[str, Any] = {
            "query": query_obj,
            "limit": k,
            "params": {"hnsw_ef": ef_search},
        }
        if with_dims_explained is not None:
            body["with_dims_explained"] = with_dims_explained
        if with_vector:
            body["with_vector"] = True
        return body

    def _parse_points(self, payload: dict[str, Any]) -> SearchResult:
        points = payload.get("result", {}).get("points", [])
        ids: list[int] = []
        scores: list[float] = []
        explained: list[dict[int, float]] = []
        vectors: list[np.ndarray] = []

        for point in points:
            ids.append(int(point["id"]))
            scores.append(float(point["score"]))
            if "dims_explained" in point:
                raw = point["dims_explained"]
                explained.append({int(k): float(v) for k, v in raw.items()})
            if "vector" in point:
                vec = point["vector"]
                if isinstance(vec, dict):
                    vec = next(iter(vec.values()))
                vectors.append(np.asarray(vec, dtype=np.float32))

        return SearchResult(
            ids=ids,
            scores=scores,
            dims_explained=explained or None,
            vectors=vectors or None,
        )

    def vanilla_search(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
        focus_masked: bool = False,
    ) -> tuple[SearchResult, int]:
        body = self._build_query_body(
            query, k, ef_search, dim_indices, focus_masked=focus_masked
        )
        start = time.perf_counter_ns()
        # Focus queries (rescore or masked) require XQdrant on xqdrant_url.
        base_url = self.xqdrant_url if dim_indices is not None else self.qdrant_url
        # [HTTP] POST /collections/{name}/points/query
        payload = self._request(
            base_url,
            "POST",
            f"/collections/{self.collection_name}/points/query",
            body,
        )
        elapsed = time.perf_counter_ns() - start
        return self._parse_points(payload), elapsed

    def post_query_attribution(
        self,
        query: np.ndarray,
        k: int,
        m: int,
        ef_search: int,
    ) -> tuple[SearchResult, int]:
        start = time.perf_counter_ns()

        # Step 1: vanilla search [HTTP]
        search_body = self._build_query_body(query, k, ef_search)
        search_payload = self._request(
            self.qdrant_url,
            "POST",
            f"/collections/{self.collection_name}/points/query",
            search_body,
        )
        base = self._parse_points(search_payload)

        # Step 2: fetch raw vectors for result IDs [HTTP]
        retrieve_payload = self._request(
            self.qdrant_url,
            "POST",
            f"/collections/{self.collection_name}/points",
            {
                "ids": base.ids,
                "with_vector": True,
            },
        )
        retrieved = retrieve_payload.get("result", [])
        id_to_vector = {}
        for item in retrieved:
            vec = item["vector"]
            if isinstance(vec, dict):
                vec = next(iter(vec.values()))
            id_to_vector[int(item["id"])] = np.asarray(vec, dtype=np.float32)

        # Step 3: client-side full sort O(D log D) per result
        explained: list[dict[int, float]] = []
        for pid in base.ids:
            vector = id_to_vector[pid]
            contribs = dot_contributions(query, vector)
            top_dims = top_m_dims_full_sort(contribs, m)
            explained.append({int(d): float(contribs[d]) for d in top_dims})

        elapsed = time.perf_counter_ns() - start
        base.dims_explained = explained
        return base, elapsed

    def xqdrant_attribution(
        self,
        query: np.ndarray,
        k: int,
        m: int,
        ef_search: int,
        dim_indices: np.ndarray | None = None,
    ) -> tuple[SearchResult, int]:
        body = self._build_query_body(
            query,
            k,
            ef_search,
            dim_indices,
            with_dims_explained={"top": m},
            with_vector=False,
        )
        start = time.perf_counter_ns()
        # [HTTP] POST /collections/{name}/points/query  (XQdrant with_dims_explained)
        payload = self._request(
            self.xqdrant_url,
            "POST",
            f"/collections/{self.collection_name}/points/query",
            body,
        )
        elapsed = time.perf_counter_ns() - start
        return self._parse_points(payload), elapsed


def make_backend(
    mode: str,
    dataset: Dataset,
    qdrant_url: str = "http://127.0.0.1:6333",
    xqdrant_url: str | None = None,
    setup_collection: bool = True,
) -> SearchBackend:
    if mode == "http":
        return HttpBackend(
            dataset,
            qdrant_url=qdrant_url,
            xqdrant_url=xqdrant_url,
            setup_collection=setup_collection,
        )
    if mode == "simulated":
        return SimulatedBackend(dataset)
    raise ValueError(f"Unknown backend mode: {mode!r}. Use 'simulated' or 'http'.")

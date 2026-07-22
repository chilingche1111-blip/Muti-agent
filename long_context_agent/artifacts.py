from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: str
    content: str
    metadata: dict[str, Any]


class ArtifactStore:
    """Thread-safe out-of-context storage shared by all graph nodes.

    Graph state only carries artifact IDs and bounded summaries. Full source text
    and intermediate products stay here and never accumulate in Supervisor state.
    """

    def __init__(self) -> None:
        self._items: dict[str, Artifact] = {}
        self._lock = threading.RLock()

    def put(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> str:
        resolved_metadata = dict(metadata or {})
        digest_source = json.dumps(
            [kind, content, resolved_metadata], ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        artifact_id = f"{kind}_{hashlib.sha256(digest_source).hexdigest()[:16]}"
        with self._lock:
            self._items[artifact_id] = Artifact(
                artifact_id=artifact_id,
                kind=kind,
                content=content,
                metadata=resolved_metadata,
            )
        return artifact_id

    def get(self, artifact_id: str) -> Artifact | None:
        with self._lock:
            return self._items.get(artifact_id)

    def describe(self, artifact_id: str) -> dict[str, Any] | None:
        artifact = self.get(artifact_id)
        if artifact is None:
            return None
        return {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "metadata": artifact.metadata,
            "characters": len(artifact.content),
        }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

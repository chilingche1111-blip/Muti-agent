from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .tokens import estimate_tokens


_CJK = re.compile(r"[\u3400-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s*")


def lexical_tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = _WORD.findall(lowered)
    chinese = "".join(_CJK.findall(lowered))
    chinese_terms = list(chinese)
    chinese_terms.extend(chinese[index : index + 2] for index in range(len(chinese) - 1))
    return words + chinese_terms


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_name: str
    index: int
    text: str
    estimated_tokens: int
    parent_id: str = ""
    section_title: str = ""
    kind: str = "raw"


@dataclass
class ParentSection:
    parent_id: str
    document_name: str
    index: int
    title: str
    text: str
    summary: str
    estimated_tokens: int
    child_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    source: str = "hybrid"


def split_text(
    text: str,
    *,
    target_tokens: int = 1_200,
    overlap_chars: int = 160,
) -> list[str]:
    """Split text on paragraph boundaries while keeping a small overlap."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    max_chars = max(1_000, int(target_tokens * 2.0))
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", normalized):
        paragraph = paragraph.strip()
        while len(paragraph) > max_chars:
            units.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars - overlap_chars :]
        if paragraph:
            units.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target_tokens:
            assembled = "\n\n".join(current).strip()
            chunks.append(assembled)
            overlap = assembled[-overlap_chars:] if overlap_chars else ""
            current = [overlap, unit] if overlap else [unit]
            current_tokens = estimate_tokens("\n\n".join(current))
        else:
            current.append(unit)
            current_tokens += unit_tokens
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def split_sections(text: str, *, fallback_parent_tokens: int = 6_000) -> list[tuple[str, str]]:
    """Preserve Markdown heading boundaries; use token-sized parents for plain text."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    matches = list(_MARKDOWN_HEADING.finditer(normalized))
    if matches:
        sections: list[tuple[str, str]] = []
        preface = normalized[: matches[0].start()].strip()
        if preface:
            sections.append(("文档前言", preface))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            title = match.group(2).strip()
            body = normalized[match.end() : end].strip()
            section_text = f"{match.group(1)} {title}\n\n{body}".strip()
            sections.append((title, section_text))
        return sections

    parents = split_text(normalized, target_tokens=fallback_parent_tokens, overlap_chars=0)
    return [(f"内容区段 {index + 1}", parent) for index, parent in enumerate(parents)]


def extractive_summary(title: str, text: str, *, max_chars: int = 900) -> str:
    """Create a deterministic, citation-safe parent summary without another model call."""
    body = _MARKDOWN_HEADING.sub("", text, count=1).strip()
    sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(body) if item.strip()]
    selected: list[str] = []
    used = 0
    for sentence in sentences:
        if selected and used + len(sentence) > max_chars:
            break
        selected.append(sentence)
        used += len(sentence)
        if len(selected) >= 3:
            break
    excerpt = " ".join(selected) or body[:max_chars]
    return f"{title}：{excerpt}"[:max_chars]


class DocumentIndex:
    """Chunked sparse retrieval index used independently by Worker agents."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.parents: dict[str, ParentSection] = {}
        self._chunk_by_id: dict[str, Chunk] = {}
        self._term_counts: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._tfidf_norms: list[float] = []
        self._parent_terms: dict[str, Counter[str]] = {}

    def add_text(self, name: str, text: str) -> list[Chunk]:
        start = len(self.chunks)
        created: list[Chunk] = []
        sections = split_sections(text)
        for parent_index, (title, section_text) in enumerate(sections):
            parent_id = f"parent_{len(self.parents):05d}"
            parent = ParentSection(
                parent_id=parent_id,
                document_name=name,
                index=parent_index,
                title=title,
                text=section_text,
                summary=extractive_summary(title, section_text),
                estimated_tokens=estimate_tokens(section_text),
            )
            for part in split_text(section_text):
                chunk_index = len(created)
                chunk = Chunk(
                    chunk_id=f"chunk_{start + chunk_index:05d}",
                    document_name=name,
                    index=chunk_index,
                    text=part,
                    estimated_tokens=estimate_tokens(part),
                    parent_id=parent_id,
                    section_title=title,
                )
                created.append(chunk)
                parent.child_ids.append(chunk.chunk_id)
            self.parents[parent_id] = parent
        self.chunks.extend(created)
        self._rebuild_statistics()
        return created

    def add_file(self, path: Path) -> list[Chunk]:
        return self.add_text(path.name, path.read_text(encoding="utf-8"))

    @property
    def hierarchy_stats(self) -> dict[str, int]:
        return {
            "raw_chunks": len(self.chunks),
            "parent_sections": len(self.parents),
            "summary_nodes": len(self.parents),
        }

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def parent_for(self, chunk: Chunk | str) -> ParentSection | None:
        resolved = self._chunk_by_id.get(chunk) if isinstance(chunk, str) else chunk
        return self.parents.get(resolved.parent_id) if resolved else None

    def _rebuild_statistics(self) -> None:
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._term_counts = [Counter(lexical_tokens(chunk.text)) for chunk in self.chunks]
        self._document_frequency = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())
        size = max(1, len(self.chunks))
        self._tfidf_norms = []
        for counts in self._term_counts:
            squared = 0.0
            for term, frequency in counts.items():
                inverse_frequency = math.log((size + 1) / (self._document_frequency[term] + 1)) + 1
                weight = (1 + math.log(frequency)) * inverse_frequency
                squared += weight * weight
            self._tfidf_norms.append(math.sqrt(squared) or 1.0)
        self._parent_terms = {
            parent_id: Counter(lexical_tokens(f"{parent.title}\n{parent.summary}"))
            for parent_id, parent in self.parents.items()
        }

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Search raw child chunks with BM25 + TF-IDF cosine + RRF."""
        query_counts = Counter(lexical_tokens(query))
        if not query_counts or not self.chunks:
            return []

        average_length = sum(sum(counts.values()) for counts in self._term_counts) / len(self.chunks)
        size = len(self.chunks)
        bm25_scores: dict[int, float] = {}
        cosine_scores: dict[int, float] = {}
        query_weights = {
            term: (1 + math.log(frequency))
            * (math.log((size + 1) / (self._document_frequency[term] + 1)) + 1)
            for term, frequency in query_counts.items()
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values())) or 1.0

        for index, counts in enumerate(self._term_counts):
            length = max(1, sum(counts.values()))
            bm25 = 0.0
            dot_product = 0.0
            for term, query_weight in query_counts.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1 + (len(self.chunks) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                saturation = frequency * 2.2 / (
                    frequency + 1.2 * (0.25 + 0.75 * length / average_length)
                )
                bm25 += query_weight * inverse_frequency * saturation
                tfidf = (1 + math.log(frequency)) * (
                    math.log((size + 1) / (document_frequency + 1)) + 1
                )
                dot_product += query_weights[term] * tfidf
            if bm25 > 0:
                bm25_scores[index] = bm25
            if dot_product > 0:
                cosine_scores[index] = dot_product / (query_norm * self._tfidf_norms[index])

        bm25_ranks = {
            index: rank
            for rank, (index, _) in enumerate(
                sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True), start=1
            )
        }
        cosine_ranks = {
            index: rank
            for rank, (index, _) in enumerate(
                sorted(cosine_scores.items(), key=lambda item: item[1], reverse=True), start=1
            )
        }
        normalized_query = query.strip().lower()
        unique_query_terms = set(query_counts)
        results: list[SearchHit] = []
        for index in bm25_scores.keys() | cosine_scores.keys():
            counts = self._term_counts[index]
            matched = sum(1 for term in unique_query_terms if counts.get(term, 0))
            coverage = matched / max(1, len(unique_query_terms))
            reciprocal_rank = 0.0
            if index in bm25_ranks:
                reciprocal_rank += 1 / (60 + bm25_ranks[index])
            if index in cosine_ranks:
                reciprocal_rank += 1 / (60 + cosine_ranks[index])
            phrase_bonus = 8.0 if normalized_query in self.chunks[index].text.lower() else 0.0
            score = reciprocal_rank * 100 + coverage * 4 + phrase_bonus
            results.append(SearchHit(chunk=self.chunks[index], score=round(score, 6)))
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def search_hierarchical(
        self,
        query: str,
        *,
        limit: int = 12,
        adjacent_children: int = 1,
        global_mode: bool = False,
    ) -> list[SearchHit]:
        """Fuse child retrieval, parent-summary matches and neighbor expansion."""
        base = self.search(query, limit=max(limit, 10))
        merged: dict[str, SearchHit] = {hit.chunk.chunk_id: hit for hit in base}
        query_terms = set(lexical_tokens(query))

        parent_scores: list[tuple[str, float]] = []
        for parent_id, counts in self._parent_terms.items():
            matched = sum(1 for term in query_terms if counts.get(term, 0))
            if not matched:
                continue
            coverage = matched / max(1, len(query_terms))
            phrase_bonus = 4.0 if query.strip().lower() in self.parents[parent_id].summary.lower() else 0.0
            parent_scores.append((parent_id, coverage * 6 + phrase_bonus))
        parent_scores.sort(key=lambda item: item[1], reverse=True)
        parent_limit = 8 if global_mode else 4
        for parent_id, parent_score in parent_scores[:parent_limit]:
            for child_id in self.parents[parent_id].child_ids[:4]:
                chunk = self._chunk_by_id[child_id]
                candidate = SearchHit(
                    chunk=chunk,
                    score=round(parent_score + (1.2 if global_mode else 0.4), 6),
                    source="parent_summary",
                )
                previous = merged.get(child_id)
                if previous is None or candidate.score > previous.score:
                    merged[child_id] = candidate

        if adjacent_children:
            for hit in list(base):
                parent = self.parent_for(hit.chunk)
                if not parent or hit.chunk.chunk_id not in parent.child_ids:
                    continue
                position = parent.child_ids.index(hit.chunk.chunk_id)
                start = max(0, position - adjacent_children)
                end = min(len(parent.child_ids), position + adjacent_children + 1)
                for distance, child_id in enumerate(parent.child_ids[start:end], start=1):
                    if child_id in merged:
                        continue
                    merged[child_id] = SearchHit(
                        chunk=self._chunk_by_id[child_id],
                        score=round(max(0.01, hit.score - 0.15 * distance), 6),
                        source="parent_neighbor",
                    )
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]

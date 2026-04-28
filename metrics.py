"""
metrics.py
==========
Coleta e armazenamento de métricas de inferência do modelo de detecção de objetos.
Suporta integração com a API de vídeos (POST /movies/import) do projeto.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class DetectionResult:
    """Resultado de uma única inferência do modelo."""

    frame_id: str
    video_id: Optional[str]                 # UUID do vídeo na API
    confidence_scores: List[float]          # Confiança de cada detecção
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    inference_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── métricas derivadas ──────────────────────────────────────────────────
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_confidence(self) -> float:
        return statistics.mean(self.confidence_scores) if self.confidence_scores else 0.0


class InferenceMetricsCollector:
    """
    Coleta e agrega métricas de inferência ao longo do tempo.

    Parâmetros
    ----------
    storage_path : str | Path, opcional
        Caminho para persistir as métricas em JSONL.
        Se None, mantém apenas em memória.
    window_size : int
        Número de amostras usadas para calcular métricas em janela deslizante.
    """

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        window_size: int = 100,
    ) -> None:
        self._records: List[DetectionResult] = []
        self._window_size = window_size
        self._storage_path = Path(storage_path) if storage_path else None

        if self._storage_path:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)

    # ── ingestão ────────────────────────────────────────────────────────────

    def add(self, result: DetectionResult) -> None:
        """Registra o resultado de uma inferência."""
        self._records.append(result)
        if self._storage_path:
            with self._storage_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(result)) + "\n")

    def add_batch(self, results: List[DetectionResult]) -> None:
        for r in results:
            self.add(r)

    def load_from_file(self, path: str | Path) -> None:
        """Carrega métricas previamente salvas em JSONL."""
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            # constrói o dataclass ignorando campos extras
            allowed = {f.name for f in DetectionResult.__dataclass_fields__.values()}
            self.add(DetectionResult(**{k: v for k, v in data.items() if k in allowed}))

    # ── agregações globais ───────────────────────────────────────────────────

    @property
    def total_samples(self) -> int:
        return len(self._records)

    def mean_confidence(self) -> float:
        vals = [r.mean_confidence for r in self._records if r.confidence_scores]
        return statistics.mean(vals) if vals else 0.0

    def mean_precision(self) -> float:
        vals = [r.precision for r in self._records]
        return statistics.mean(vals) if vals else 0.0

    def mean_recall(self) -> float:
        vals = [r.recall for r in self._records]
        return statistics.mean(vals) if vals else 0.0

    def mean_f1(self) -> float:
        vals = [r.f1_score for r in self._records]
        return statistics.mean(vals) if vals else 0.0

    def false_positive_rate(self) -> float:
        total_fp = sum(r.false_positives for r in self._records)
        total_det = sum(r.true_positives + r.false_positives for r in self._records)
        return total_fp / total_det if total_det > 0 else 0.0

    def false_negative_rate(self) -> float:
        total_fn = sum(r.false_negatives for r in self._records)
        total_real = sum(r.true_positives + r.false_negatives for r in self._records)
        return total_fn / total_real if total_real > 0 else 0.0

    def mean_inference_time(self) -> float:
        vals = [r.inference_time_ms for r in self._records if r.inference_time_ms > 0]
        return statistics.mean(vals) if vals else 0.0

    # ── janela deslizante ────────────────────────────────────────────────────

    def windowed_metrics(self, n: Optional[int] = None) -> dict:
        """Retorna métricas apenas das últimas *n* amostras."""
        n = n or self._window_size
        window = self._records[-n:]
        if not window:
            return {}

        conf_vals = [r.mean_confidence for r in window if r.confidence_scores]
        return {
            "window_size": len(window),
            "mean_confidence": statistics.mean(conf_vals) if conf_vals else 0.0,
            "mean_precision": statistics.mean([r.precision for r in window]),
            "mean_recall": statistics.mean([r.recall for r in window]),
            "mean_f1": statistics.mean([r.f1_score for r in window]),
            "false_positive_rate": (
                sum(r.false_positives for r in window)
                / max(sum(r.true_positives + r.false_positives for r in window), 1)
            ),
        }

    # ── estabilidade temporal ────────────────────────────────────────────────

    def confidence_trend(self, n_bins: int = 5) -> List[float]:
        """
        Divide as amostras em *n_bins* fatias temporais e retorna
        a confiança média de cada fatia. Útil para visualizar degradação.
        """
        if not self._records:
            return []
        chunk = max(1, len(self._records) // n_bins)
        return [
            statistics.mean(
                r.mean_confidence
                for r in self._records[i : i + chunk]
                if r.confidence_scores
            )
            for i in range(0, len(self._records), chunk)
        ]

    def confidence_std(self) -> float:
        vals = [r.mean_confidence for r in self._records if r.confidence_scores]
        return statistics.pstdev(vals) if len(vals) > 1 else 0.0

    def to_summary_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "mean_confidence": round(self.mean_confidence(), 4),
            "mean_precision": round(self.mean_precision(), 4),
            "mean_recall": round(self.mean_recall(), 4),
            "mean_f1": round(self.mean_f1(), 4),
            "false_positive_rate": round(self.false_positive_rate(), 4),
            "false_negative_rate": round(self.false_negative_rate(), 4),
            "mean_inference_time_ms": round(self.mean_inference_time(), 2),
            "confidence_std": round(self.confidence_std(), 4),
        }

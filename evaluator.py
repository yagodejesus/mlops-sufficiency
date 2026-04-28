"""
evaluator.py
============
Avaliador principal de suficiência de treinamento.
Combina métricas de inferência, detecção de deriva e critérios
personalizados para decidir se o modelo já está "treinado o suficiente".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .drift import DataDriftDetector, DriftReport, DriftSeverity
from .metrics import InferenceMetricsCollector


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "sufficient"           # modelo pronto para produção
    NEEDS_MORE_DATA = "needs_more_data" # poucos dados para decidir
    NEEDS_RETRAINING = "needs_retraining"  # treinamento adicional necessário
    DEGRADED = "degraded"               # em produção, mas degradado


@dataclass
class SufficiencyThresholds:
    """
    Limiares que definem o que é "suficiente" para o modelo de
    detecção de resíduos urbanos do projeto de mestrado.

    Ajuste conforme os resultados experimentais (seção 4.2 do plano).
    """

    # ── mínimos absolutos ─────────────────────────────────────────────────
    min_mean_confidence: float = 0.70       # confiança mínima média
    min_mean_precision: float = 0.75        # precisão mínima
    min_mean_recall: float = 0.70           # recall mínimo
    min_mean_f1: float = 0.72              # F1 mínimo

    # ── máximos aceitáveis ────────────────────────────────────────────────
    max_false_positive_rate: float = 0.15   # FP rate máximo
    max_false_negative_rate: float = 0.20   # FN rate máximo
    max_inference_time_ms: float = 500.0    # latência máxima em ms

    # ── estabilidade ──────────────────────────────────────────────────────
    max_confidence_std: float = 0.12        # volatilidade máxima da confiança
    min_confidence_trend_stability: float = 0.02  # queda máxima entre fatias temporais

    # ── volume mínimo de dados ────────────────────────────────────────────
    min_samples: int = 200                  # amostras mínimas para avaliação


@dataclass
class SufficiencyResult:
    status: SufficiencyStatus
    score: float                            # 0.0 – 1.0 (suficiência geral)
    passed_criteria: List[str] = field(default_factory=list)
    failed_criteria: List[str] = field(default_factory=list)
    drift_report: Optional[DriftReport] = None
    metrics_summary: Dict = field(default_factory=dict)
    recommendation: str = ""

    def is_sufficient(self) -> bool:
        return self.status == SufficiencyStatus.SUFFICIENT

    def needs_retraining(self) -> bool:
        return self.status in (
            SufficiencyStatus.NEEDS_RETRAINING,
            SufficiencyStatus.DEGRADED,
        )


class TrainingSufficiencyEvaluator:
    """
    Avalia se o modelo de detecção de objetos já foi treinado o suficiente.

    Combina:
    - Critérios absolutos de qualidade (precisão, recall, F1, confiança)
    - Estabilidade temporal (tendência e volatilidade)
    - Detecção de deriva (quando há baseline de produção)

    Exemplo de uso
    --------------
    >>> from mlops_sufficiency import (
    ...     TrainingSufficiencyEvaluator,
    ...     InferenceMetricsCollector,
    ...     DetectionResult,
    ... )
    >>> collector = InferenceMetricsCollector()
    >>> # ... adicione resultados de inferência ...
    >>> evaluator = TrainingSufficiencyEvaluator()
    >>> result = evaluator.evaluate(collector)
    >>> print(result.status, result.score)
    """

    def __init__(
        self,
        thresholds: Optional[SufficiencyThresholds] = None,
        drift_detector: Optional[DataDriftDetector] = None,
    ) -> None:
        self.thresholds = thresholds or SufficiencyThresholds()
        self.drift_detector = drift_detector

    # ── avaliação principal ────────────────────────────────────────────────

    def evaluate(
        self, collector: InferenceMetricsCollector
    ) -> SufficiencyResult:
        """
        Avalia a suficiência do treinamento com base nas métricas coletadas.

        Parâmetros
        ----------
        collector : InferenceMetricsCollector
            Coletor com as métricas de inferência do modelo atual.

        Retorna
        -------
        SufficiencyResult com status, score e recomendações detalhadas.
        """
        t = self.thresholds
        passed: List[str] = []
        failed: List[str] = []

        # ── 1. volume mínimo de dados ──────────────────────────────────────
        if collector.total_samples < t.min_samples:
            return SufficiencyResult(
                status=SufficiencyStatus.NEEDS_MORE_DATA,
                score=0.0,
                failed_criteria=[
                    f"Amostras insuficientes: {collector.total_samples} / {t.min_samples} mínimo"
                ],
                metrics_summary=collector.to_summary_dict(),
                recommendation=(
                    f"Colete pelo menos {t.min_samples} amostras antes de avaliar suficiência. "
                    f"Processe mais vídeos pela API (/movies/import) e execute inferência."
                ),
            )

        summary = collector.to_summary_dict()

        # ── 2. critérios absolutos de qualidade ───────────────────────────
        checks = [
            (
                summary["mean_confidence"] >= t.min_mean_confidence,
                f"Confiança média {summary['mean_confidence']:.3f} >= {t.min_mean_confidence}",
                f"Confiança média {summary['mean_confidence']:.3f} < {t.min_mean_confidence} (mínimo)",
            ),
            (
                summary["mean_precision"] >= t.min_mean_precision,
                f"Precisão {summary['mean_precision']:.3f} >= {t.min_mean_precision}",
                f"Precisão {summary['mean_precision']:.3f} < {t.min_mean_precision} (mínimo)",
            ),
            (
                summary["mean_recall"] >= t.min_mean_recall,
                f"Recall {summary['mean_recall']:.3f} >= {t.min_mean_recall}",
                f"Recall {summary['mean_recall']:.3f} < {t.min_mean_recall} (mínimo)",
            ),
            (
                summary["mean_f1"] >= t.min_mean_f1,
                f"F1 {summary['mean_f1']:.3f} >= {t.min_mean_f1}",
                f"F1 {summary['mean_f1']:.3f} < {t.min_mean_f1} (mínimo)",
            ),
            (
                summary["false_positive_rate"] <= t.max_false_positive_rate,
                f"Taxa FP {summary['false_positive_rate']:.3f} <= {t.max_false_positive_rate}",
                f"Taxa FP {summary['false_positive_rate']:.3f} > {t.max_false_positive_rate} (máximo)",
            ),
            (
                summary["false_negative_rate"] <= t.max_false_negative_rate,
                f"Taxa FN {summary['false_negative_rate']:.3f} <= {t.max_false_negative_rate}",
                f"Taxa FN {summary['false_negative_rate']:.3f} > {t.max_false_negative_rate} (máximo)",
            ),
        ]

        if summary["mean_inference_time_ms"] > 0:
            checks.append((
                summary["mean_inference_time_ms"] <= t.max_inference_time_ms,
                f"Latência {summary['mean_inference_time_ms']:.1f}ms <= {t.max_inference_time_ms}ms",
                f"Latência {summary['mean_inference_time_ms']:.1f}ms > {t.max_inference_time_ms}ms (máximo)",
            ))

        for ok, msg_pass, msg_fail in checks:
            (passed if ok else failed).append(ok and msg_pass or msg_fail)

        # ── 3. estabilidade temporal ───────────────────────────────────────
        trend = collector.confidence_trend(n_bins=5)
        if len(trend) >= 2:
            max_drop = max(trend[i] - trend[i + 1] for i in range(len(trend) - 1))
            if max_drop <= t.min_confidence_trend_stability:
                passed.append(f"Tendência estável (queda máx. {max_drop:.3f})")
            else:
                failed.append(
                    f"Queda de confiança entre janelas temporais: {max_drop:.3f} "
                    f"> {t.min_confidence_trend_stability} (máximo)"
                )

        volatility = summary["confidence_std"]
        if volatility <= t.max_confidence_std:
            passed.append(f"Volatilidade aceitável (std={volatility:.3f})")
        else:
            failed.append(
                f"Alta volatilidade: std={volatility:.3f} > {t.max_confidence_std}"
            )

        # ── 4. detecção de deriva (opcional) ──────────────────────────────
        drift_report: Optional[DriftReport] = None
        if self.drift_detector is not None:
            try:
                drift_report = self.drift_detector.evaluate(collector)
                if drift_report.severity == DriftSeverity.NONE:
                    passed.append("Sem deriva de dados detectada")
                elif drift_report.severity == DriftSeverity.LOW:
                    passed.append("Deriva leve (dentro do tolerável)")
                else:
                    for rule in drift_report.triggered_rules:
                        failed.append(f"[Drift] {rule}")
            except RuntimeError:
                pass  # baseline não configurado ainda

        # ── 5. score e status final ────────────────────────────────────────
        total_criteria = len(passed) + len(failed)
        score = len(passed) / total_criteria if total_criteria > 0 else 0.0

        if drift_report and drift_report.severity == DriftSeverity.HIGH:
            status = SufficiencyStatus.DEGRADED
        elif score >= 0.85 and len(failed) == 0:
            status = SufficiencyStatus.SUFFICIENT
        elif score >= 0.85:
            status = SufficiencyStatus.SUFFICIENT
        elif drift_report and drift_report.is_retraining_needed():
            status = SufficiencyStatus.NEEDS_RETRAINING
        elif len(failed) > len(passed):
            status = SufficiencyStatus.NEEDS_RETRAINING
        else:
            status = SufficiencyStatus.NEEDS_RETRAINING

        recommendation = self._build_recommendation(status, failed, drift_report)

        return SufficiencyResult(
            status=status,
            score=round(score, 4),
            passed_criteria=passed,
            failed_criteria=failed,
            drift_report=drift_report,
            metrics_summary=summary,
            recommendation=recommendation,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_recommendation(
        self,
        status: SufficiencyStatus,
        failed: List[str],
        drift: Optional[DriftReport],
    ) -> str:
        if status == SufficiencyStatus.SUFFICIENT:
            return (
                "✅ Modelo suficientemente treinado. Pode ser promovido para produção. "
                "Mantenha o monitoramento contínuo via InferenceMetricsCollector."
            )
        if status == SufficiencyStatus.DEGRADED:
            return (
                "🔴 Modelo degradado em produção. Re-treinamento urgente. "
                "Revise os vídeos processados recentemente pelo sistema (status 'Processado') "
                "e avalie a qualidade das detecções."
            )
        if status == SufficiencyStatus.NEEDS_RETRAINING:
            tips = []
            for f in failed:
                if "Confiança" in f or "F1" in f or "Recall" in f:
                    tips.append(
                        "- Aumentar diversidade do dataset com novos vídeos urbanos"
                    )
                if "FP" in f:
                    tips.append(
                        "- Revisar anotações de falsos positivos e adicionar hard-negatives"
                    )
                if "FN" in f:
                    tips.append(
                        "- Aumentar exemplos de resíduos de difícil detecção (pequenos, ocluídos)"
                    )
                if "Latência" in f:
                    tips.append(
                        "- Considerar quantização ou pruning do modelo para reduzir latência"
                    )
            unique_tips = list(dict.fromkeys(tips))
            return (
                "🟡 Re-treinamento necessário. Sugestões:\n"
                + "\n".join(unique_tips or ["- Revisar dataset e hiperparâmetros de treinamento"])
            )
        return "⚪ Dados insuficientes para avaliação."

    # ── avaliação rápida (sem instância) ──────────────────────────────────────

    @classmethod
    def quick_check(
        cls,
        collector: InferenceMetricsCollector,
        **threshold_overrides,
    ) -> SufficiencyResult:
        """
        Avaliação rápida sem instanciar o avaliador explicitamente.

        Exemplo
        -------
        >>> result = TrainingSufficiencyEvaluator.quick_check(
        ...     collector, min_mean_f1=0.80
        ... )
        """
        thresholds = SufficiencyThresholds(**threshold_overrides)
        return cls(thresholds=thresholds).evaluate(collector)

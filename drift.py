"""
drift.py
========
Detecção de deriva de dados (data drift) e degradação de desempenho.
Implementa os critérios definidos no plano de mestrado (seção 4.2).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .metrics import InferenceMetricsCollector


class DriftSeverity(str, Enum):
    NONE = "none"           # sem deriva detectada
    LOW = "low"             # deriva leve – monitorar
    MEDIUM = "medium"       # deriva moderada – alerta
    HIGH = "high"           # deriva severa – re-treinar


@dataclass
class DriftReport:
    severity: DriftSeverity
    confidence_drop: float          # queda absoluta na confiança (janela vs base)
    f1_drop: float                  # queda absoluta no F1
    fp_rate_increase: float         # aumento na taxa de falsos positivos
    fn_rate_increase: float         # aumento na taxa de falsos negativos
    confidence_volatility: float    # desvio padrão da confiança
    triggered_rules: List[str] = field(default_factory=list)
    recommendation: str = ""

    def is_retraining_needed(self) -> bool:
        return self.severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH)


class DataDriftDetector:
    """
    Detecta deriva de dados comparando métricas da janela atual
    com uma linha de base (baseline) estabelecida após treinamento.

    Parâmetros
    ----------
    confidence_drop_threshold : float
        Queda aceitável na confiança antes de alertar (padrão 0.05 = 5%).
    f1_drop_threshold : float
        Queda aceitável no F1 antes de alertar (padrão 0.05).
    fp_rate_increase_threshold : float
        Aumento aceitável na taxa de falsos positivos (padrão 0.05).
    fn_rate_increase_threshold : float
        Aumento aceitável na taxa de falsos negativos (padrão 0.05).
    confidence_volatility_threshold : float
        Desvio padrão máximo aceitável da confiança (padrão 0.15).
    window_size : int
        Tamanho da janela deslizante para avaliação (padrão 100).
    """

    def __init__(
        self,
        confidence_drop_threshold: float = 0.05,
        f1_drop_threshold: float = 0.05,
        fp_rate_increase_threshold: float = 0.05,
        fn_rate_increase_threshold: float = 0.05,
        confidence_volatility_threshold: float = 0.15,
        window_size: int = 100,
    ) -> None:
        self.confidence_drop_threshold = confidence_drop_threshold
        self.f1_drop_threshold = f1_drop_threshold
        self.fp_rate_increase_threshold = fp_rate_increase_threshold
        self.fn_rate_increase_threshold = fn_rate_increase_threshold
        self.confidence_volatility_threshold = confidence_volatility_threshold
        self.window_size = window_size

        # baseline – estabelecido manualmente ou pela primeira janela
        self._baseline: Optional[dict] = None

    # ── baseline ──────────────────────────────────────────────────────────────

    def set_baseline(
        self,
        mean_confidence: float,
        mean_f1: float,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
    ) -> None:
        """Define os valores de referência esperados do modelo treinado."""
        self._baseline = {
            "mean_confidence": mean_confidence,
            "mean_f1": mean_f1,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }

    def set_baseline_from_collector(
        self, collector: InferenceMetricsCollector
    ) -> None:
        """Deriva o baseline das métricas atuais do coletor (snapshot inicial)."""
        self._baseline = {
            "mean_confidence": collector.mean_confidence(),
            "mean_f1": collector.mean_f1(),
            "false_positive_rate": collector.false_positive_rate(),
            "false_negative_rate": collector.false_negative_rate(),
        }

    # ── detecção ──────────────────────────────────────────────────────────────

    def evaluate(self, collector: InferenceMetricsCollector) -> DriftReport:
        """
        Avalia se há deriva de dados comparando a janela atual com o baseline.

        Retorna
        -------
        DriftReport com severidade e recomendação de ação.
        """
        if not self._baseline:
            raise RuntimeError(
                "Baseline não definido. Chame set_baseline() ou "
                "set_baseline_from_collector() antes de avaliar."
            )

        window = collector.windowed_metrics(self.window_size)
        if not window:
            raise ValueError("Sem dados suficientes para avaliação.")

        # ── deltas ────────────────────────────────────────────────────────────
        conf_drop = self._baseline["mean_confidence"] - window["mean_confidence"]
        f1_drop = self._baseline["mean_f1"] - window["mean_f1"]
        fp_increase = window["false_positive_rate"] - self._baseline["false_positive_rate"]
        fn_increase = window.get("false_negative_rate", 0.0) - self._baseline["false_negative_rate"]
        volatility = collector.confidence_std()

        triggered_rules: List[str] = []

        if conf_drop > self.confidence_drop_threshold:
            triggered_rules.append(
                f"Confiança caiu {conf_drop:.2%} (limite: {self.confidence_drop_threshold:.2%})"
            )
        if f1_drop > self.f1_drop_threshold:
            triggered_rules.append(
                f"F1 caiu {f1_drop:.2%} (limite: {self.f1_drop_threshold:.2%})"
            )
        if fp_increase > self.fp_rate_increase_threshold:
            triggered_rules.append(
                f"Taxa de falsos positivos aumentou {fp_increase:.2%}"
            )
        if fn_increase > self.fn_rate_increase_threshold:
            triggered_rules.append(
                f"Taxa de falsos negativos aumentou {fn_increase:.2%}"
            )
        if volatility > self.confidence_volatility_threshold:
            triggered_rules.append(
                f"Alta volatilidade na confiança: std={volatility:.3f}"
            )

        # ── classificação de severidade ────────────────────────────────────────
        n = len(triggered_rules)
        if n == 0:
            severity = DriftSeverity.NONE
            recommendation = (
                "Modelo estável. Nenhuma ação necessária. "
                "Continuar monitoramento regular."
            )
        elif n == 1:
            severity = DriftSeverity.LOW
            recommendation = (
                "Deriva leve detectada. Aumentar frequência de monitoramento. "
                "Coletar mais amostras do ambiente real."
            )
        elif n == 2:
            severity = DriftSeverity.MEDIUM
            recommendation = (
                "Deriva moderada detectada. Iniciar coleta intensiva de dados reais. "
                "Agendar ciclo de re-treinamento em breve."
            )
        else:
            severity = DriftSeverity.HIGH
            recommendation = (
                "Deriva severa detectada. Re-treinamento URGENTE recomendado. "
                "Verificar qualidade do pipeline de dados e condições dos vídeos "
                "processados pelo sistema."
            )

        return DriftReport(
            severity=severity,
            confidence_drop=max(conf_drop, 0.0),
            f1_drop=max(f1_drop, 0.0),
            fp_rate_increase=max(fp_increase, 0.0),
            fn_rate_increase=max(fn_increase, 0.0),
            confidence_volatility=volatility,
            triggered_rules=triggered_rules,
            recommendation=recommendation,
        )

    # ── utilitário: KS-test simplificado (sem scipy) ─────────────────────────

    @staticmethod
    def kolmogorov_smirnov_distance(
        sample_a: List[float], sample_b: List[float]
    ) -> float:
        """
        Calcula a distância KS entre duas distribuições empíricas.
        Útil para comparar distribuições de confiança entre janelas temporais.
        Implementação pura Python – não requer scipy.
        """
        if not sample_a or not sample_b:
            return 0.0
        all_vals = sorted(set(sample_a + sample_b))
        n_a, n_b = len(sample_a), len(sample_b)
        sorted_a = sorted(sample_a)
        sorted_b = sorted(sample_b)

        def ecdf(sorted_data: List[float], x: float) -> float:
            lo, hi = 0, len(sorted_data)
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_data[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo / len(sorted_data)

        return max(abs(ecdf(sorted_a, v) - ecdf(sorted_b, v)) for v in all_vals)

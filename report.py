"""
report.py
=========
Gerador de relatórios de suficiência de treinamento.
Produz saída formatada em texto (para logs/CLI) e em JSON (para integração).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .evaluator import SufficiencyResult, SufficiencyStatus
from .drift import DriftSeverity


_SEVERITY_EMOJI = {
    DriftSeverity.NONE: "🟢",
    DriftSeverity.LOW: "🟡",
    DriftSeverity.MEDIUM: "🟠",
    DriftSeverity.HIGH: "🔴",
}

_STATUS_EMOJI = {
    SufficiencyStatus.SUFFICIENT: "✅",
    SufficiencyStatus.NEEDS_MORE_DATA: "⚪",
    SufficiencyStatus.NEEDS_RETRAINING: "🟡",
    SufficiencyStatus.DEGRADED: "🔴",
}


class SufficiencyReport:
    """
    Renderiza um SufficiencyResult em diferentes formatos.

    Parâmetros
    ----------
    result : SufficiencyResult
        Resultado retornado por TrainingSufficiencyEvaluator.evaluate().
    model_name : str
        Nome/versão do modelo avaliado (para o cabeçalho do relatório).
    """

    def __init__(
        self,
        result: SufficiencyResult,
        model_name: str = "modelo-deteccao-residuos",
    ) -> None:
        self.result = result
        self.model_name = model_name
        self._generated_at = datetime.utcnow().isoformat() + "Z"

    # ── texto ─────────────────────────────────────────────────────────────────

    def as_text(self) -> str:
        r = self.result
        status_emoji = _STATUS_EMOJI.get(r.status, "❓")
        lines = [
            "=" * 62,
            "  RELATÓRIO DE SUFICIÊNCIA DE TREINAMENTO  —  MLOps",
            "=" * 62,
            f"  Modelo       : {self.model_name}",
            f"  Avaliado em  : {self._generated_at}",
            f"  Status       : {status_emoji} {r.status.value.upper()}",
            f"  Score geral  : {r.score:.1%}",
            "-" * 62,
            "  MÉTRICAS AGREGADAS",
            "-" * 62,
        ]

        m = r.metrics_summary
        for key, label in [
            ("total_samples", "Amostras totais"),
            ("mean_confidence", "Confiança média"),
            ("mean_precision", "Precisão"),
            ("mean_recall", "Recall"),
            ("mean_f1", "F1-score"),
            ("false_positive_rate", "Taxa FP"),
            ("false_negative_rate", "Taxa FN"),
            ("mean_inference_time_ms", "Latência média (ms)"),
            ("confidence_std", "Volatilidade (std conf.)"),
        ]:
            val = m.get(key, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            lines.append(f"  {label:<28}: {val}")

        if r.passed_criteria:
            lines += ["-" * 62, "  CRITÉRIOS APROVADOS"]
            for c in r.passed_criteria:
                lines.append(f"    ✓ {c}")

        if r.failed_criteria:
            lines += ["-" * 62, "  CRITÉRIOS REPROVADOS"]
            for c in r.failed_criteria:
                lines.append(f"    ✗ {c}")

        if r.drift_report:
            d = r.drift_report
            sev_emoji = _SEVERITY_EMOJI.get(d.severity, "❓")
            lines += [
                "-" * 62,
                "  ANÁLISE DE DERIVA (DATA DRIFT)",
                f"  Severidade          : {sev_emoji} {d.severity.value.upper()}",
                f"  Queda de confiança  : {d.confidence_drop:.4f}",
                f"  Queda de F1         : {d.f1_drop:.4f}",
                f"  Aumento de FP       : {d.fp_rate_increase:.4f}",
                f"  Aumento de FN       : {d.fn_rate_increase:.4f}",
                f"  Volatilidade conf.  : {d.confidence_volatility:.4f}",
            ]
            if d.triggered_rules:
                lines.append("  Regras ativadas:")
                for rule in d.triggered_rules:
                    lines.append(f"    • {rule}")

        lines += [
            "-" * 62,
            "  RECOMENDAÇÃO",
            "",
        ]
        for line in r.recommendation.splitlines():
            lines.append(f"  {line}")

        lines.append("=" * 62)
        return "\n".join(lines)

    # ── JSON ──────────────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        r = self.result
        payload = {
            "model_name": self.model_name,
            "generated_at": self._generated_at,
            "status": r.status.value,
            "score": r.score,
            "is_sufficient": r.is_sufficient(),
            "needs_retraining": r.needs_retraining(),
            "metrics": r.metrics_summary,
            "passed_criteria": r.passed_criteria,
            "failed_criteria": r.failed_criteria,
            "recommendation": r.recommendation,
        }
        if r.drift_report:
            d = r.drift_report
            payload["drift"] = {
                "severity": d.severity.value,
                "confidence_drop": d.confidence_drop,
                "f1_drop": d.f1_drop,
                "fp_rate_increase": d.fp_rate_increase,
                "fn_rate_increase": d.fn_rate_increase,
                "confidence_volatility": d.confidence_volatility,
                "triggered_rules": d.triggered_rules,
                "retraining_needed": d.is_retraining_needed(),
            }
        return payload

    def as_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent)

    # ── salvar em disco ───────────────────────────────────────────────────────

    def save(self, output_dir: str = ".", fmt: str = "both") -> None:
        """
        Salva o relatório em disco.

        Parâmetros
        ----------
        output_dir : str
            Diretório de saída.
        fmt : str
            "text", "json" ou "both".
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = self._generated_at[:19].replace(":", "-")
        stem = f"sufficiency_{self.model_name}_{ts}"

        if fmt in ("text", "both"):
            (out / f"{stem}.txt").write_text(self.as_text(), encoding="utf-8")
        if fmt in ("json", "both"):
            (out / f"{stem}.json").write_text(self.as_json(), encoding="utf-8")

    # ── print direto ──────────────────────────────────────────────────────────

    def print(self) -> None:
        print(self.as_text())

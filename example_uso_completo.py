"""
example_uso_completo.py
=======================
Exemplo completo de uso da biblioteca mlops_sufficiency
aplicado ao projeto de mestrado: detecção de resíduos urbanos.

Simula o pipeline real do sistema:
  1. Vídeos são enviados via API (/movies/import)
  2. Modelo YOLO processa os frames
  3. Métricas são coletadas por InferenceMetricsCollector
  4. TrainingSufficiencyEvaluator decide se precisa re-treinar
  5. SufficiencyReport gera o relatório para o orientador / sistema CI
"""

import random
import sys
import os

# Adiciona o diretório pai ao path para importar o pacote local
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops_sufficiency import (
    TrainingSufficiencyEvaluator,
    InferenceMetricsCollector,
    DataDriftDetector,
    SufficiencyReport,
)
from mlops_sufficiency.metrics import DetectionResult
from mlops_sufficiency.evaluator import SufficiencyThresholds


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de simulação
# ─────────────────────────────────────────────────────────────────────────────

def simular_inferencia(
    n: int,
    mean_conf: float = 0.78,
    std_conf: float = 0.08,
    tp_base: int = 8,
    fp_base: int = 1,
    fn_base: int = 2,
    seed: int = 42,
) -> list:
    """
    Simula *n* resultados de inferência do modelo YOLO sobre frames de vídeos
    urbanos com resíduos.
    """
    random.seed(seed)
    results = []
    for i in range(n):
        n_det = random.randint(1, 12)
        scores = [
            max(0.0, min(1.0, random.gauss(mean_conf, std_conf)))
            for _ in range(n_det)
        ]
        results.append(DetectionResult(
            frame_id=f"frame_{i:05d}",
            video_id=f"video-uuid-{i // 30:04d}",  # ~30 frames por vídeo
            confidence_scores=scores,
            true_positives=tp_base + random.randint(-2, 2),
            false_positives=fp_base + random.randint(0, 2),
            false_negatives=fn_base + random.randint(0, 1),
            inference_time_ms=random.uniform(80, 200),
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CENÁRIO 1 — Modelo bem treinado (suficiente para produção)
# ─────────────────────────────────────────────────────────────────────────────

def cenario_modelo_bom():
    print("\n" + "█" * 62)
    print("  CENÁRIO 1: Modelo bem treinado")
    print("█" * 62)

    collector = InferenceMetricsCollector(window_size=100)
    collector.add_batch(simular_inferencia(
        n=300,
        mean_conf=0.82,
        std_conf=0.06,
        tp_base=9,
        fp_base=1,
        fn_base=1,
    ))

    thresholds = SufficiencyThresholds(
        min_mean_confidence=0.70,
        min_mean_f1=0.72,
        min_samples=200,
    )
    evaluator = TrainingSufficiencyEvaluator(thresholds=thresholds)
    result = evaluator.evaluate(collector)

    report = SufficiencyReport(result, model_name="yolov8-residuos-v1.2")
    report.print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CENÁRIO 2 — Modelo insuficiente (precisa de mais treinamento)
# ─────────────────────────────────────────────────────────────────────────────

def cenario_modelo_insuficiente():
    print("\n" + "█" * 62)
    print("  CENÁRIO 2: Modelo insuficiente — precisa re-treinar")
    print("█" * 62)

    collector = InferenceMetricsCollector(window_size=100)
    collector.add_batch(simular_inferencia(
        n=250,
        mean_conf=0.55,   # confiança baixa
        std_conf=0.18,    # alta volatilidade
        tp_base=5,
        fp_base=4,        # muitos falsos positivos
        fn_base=5,        # muitos falsos negativos
        seed=99,
    ))

    evaluator = TrainingSufficiencyEvaluator()
    result = evaluator.evaluate(collector)

    report = SufficiencyReport(result, model_name="yolov8-residuos-v0.3-draft")
    report.print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CENÁRIO 3 — Modelo em produção com deriva detectada
# ─────────────────────────────────────────────────────────────────────────────

def cenario_deriva_dados():
    print("\n" + "█" * 62)
    print("  CENÁRIO 3: Modelo em produção com deriva de dados")
    print("█" * 62)

    # Fase 1: métricas iniciais (após deploy) — usadas como baseline
    collector_baseline = InferenceMetricsCollector()
    collector_baseline.add_batch(simular_inferencia(
        n=200, mean_conf=0.80, std_conf=0.07, tp_base=9, fp_base=1, fn_base=1, seed=1
    ))

    # Configura detector de deriva com o baseline da fase 1
    drift_detector = DataDriftDetector(
        confidence_drop_threshold=0.05,
        f1_drop_threshold=0.05,
        fp_rate_increase_threshold=0.05,
        fn_rate_increase_threshold=0.05,
        confidence_volatility_threshold=0.12,
        window_size=100,
    )
    drift_detector.set_baseline_from_collector(collector_baseline)

    # Fase 2: novos dados reais chegam — cenário urbano diferente (noite, chuva)
    # Simula degradação gradual
    collector_producao = InferenceMetricsCollector(window_size=100)
    collector_producao.add_batch(simular_inferencia(
        n=150, mean_conf=0.80, std_conf=0.07, seed=1  # início OK
    ))
    collector_producao.add_batch(simular_inferencia(
        n=150,
        mean_conf=0.61,   # queda de confiança (cenário noturno)
        std_conf=0.15,    # mais incerto
        tp_base=6,
        fp_base=3,
        fn_base=4,
        seed=77,
    ))

    evaluator = TrainingSufficiencyEvaluator(
        drift_detector=drift_detector,
    )
    result = evaluator.evaluate(collector_producao)

    report = SufficiencyReport(result, model_name="yolov8-residuos-v1.2-producao")
    report.print()

    # Exibe relatório JSON também
    print("\n📄 JSON para integração com pipeline CI/CD:")
    print(report.as_json())

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CENÁRIO 4 — Poucos dados (não é possível avaliar ainda)
# ─────────────────────────────────────────────────────────────────────────────

def cenario_dados_insuficientes():
    print("\n" + "█" * 62)
    print("  CENÁRIO 4: Dados insuficientes para avaliar")
    print("█" * 62)

    collector = InferenceMetricsCollector()
    collector.add_batch(simular_inferencia(n=50))  # menos que o mínimo de 200

    result = TrainingSufficiencyEvaluator.quick_check(collector)
    report = SufficiencyReport(result, model_name="yolov8-residuos-experimento")
    report.print()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Demonstração da distância KS (comparação de distribuições)
# ─────────────────────────────────────────────────────────────────────────────

def demo_ks_distance():
    print("\n" + "─" * 62)
    print("  DEMO: Distância de Kolmogorov-Smirnov entre janelas")
    print("─" * 62)

    random.seed(42)
    janela_inicial = [random.gauss(0.80, 0.07) for _ in range(200)]
    janela_atual   = [random.gauss(0.63, 0.14) for _ in range(200)]

    ks = DataDriftDetector.kolmogorov_smirnov_distance(janela_inicial, janela_atual)
    print(f"  KS distance: {ks:.4f}")
    print(
        f"  Interpretação: {'⚠️  Deriva detectada' if ks > 0.1 else '✅ Distribuições similares'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  mlops_sufficiency — Exemplo de Uso Completo")
    print("  Projeto de Mestrado: Yago de Jesus Martins")
    print("  Detecção de Resíduos Urbanos com MLOps")
    print("=" * 62)

    r1 = cenario_modelo_bom()
    r2 = cenario_modelo_insuficiente()
    r3 = cenario_deriva_dados()
    r4 = cenario_dados_insuficientes()
    demo_ks_distance()

    print("\n" + "=" * 62)
    print("  RESUMO DOS CENÁRIOS")
    print("=" * 62)
    for nome, resultado in [
        ("Modelo bom (v1.2)", r1),
        ("Modelo insuficiente (v0.3)", r2),
        ("Deriva em produção (v1.2)", r3),
        ("Dados insuficientes", r4),
    ]:
        emoji = "✅" if resultado.is_sufficient() else ("🔴" if resultado.needs_retraining() else "⚪")
        print(f"  {emoji} {nome:<35} status={resultado.status.value:<20} score={resultado.score:.1%}")
    print("=" * 62)

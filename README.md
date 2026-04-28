# mlops_sufficiency

> **Biblioteca Python para avaliação de suficiência de treinamento de modelos de IA**  
> Desenvolvida para o projeto de mestrado em MLOps aplicado à detecção de resíduos urbanos.  
> Autor: Yago de Jesus Martins

---

## Visão Geral

`mlops_sufficiency` responde à pergunta central do projeto de mestrado:

> **"O modelo de detecção de objetos já está treinado o suficiente?"**

A biblioteca avalia continuamente as métricas de inferência do modelo e decide, com base em critérios configuráveis, se o modelo está pronto para produção, se precisa de mais treinamento ou se está degradando em ambiente real.

---

## Módulos

| Módulo | Responsabilidade |
|---|---|
| `InferenceMetricsCollector` | Coleta e agrega métricas de cada frame inferido |
| `DataDriftDetector` | Detecta deriva de dados comparando janelas temporais com o baseline |
| `TrainingSufficiencyEvaluator` | Decide o status de suficiência com score e critérios detalhados |
| `SufficiencyReport` | Gera relatórios em texto e JSON para logs, CI/CD e orientadores |
| `VideoAPIClient` | Integração com a API de vídeos do sistema (`/movies/import`) |

---

## Instalação

```bash
# Clonando o repositório
git clone <repo>
cd mlops_sufficiency
pip install -e .

# Sem dependências externas — funciona com Python 3.9+ puro
```

---

## Uso Básico

```python
from mlops_sufficiency import (
    TrainingSufficiencyEvaluator,
    InferenceMetricsCollector,
    SufficiencyReport,
)
from mlops_sufficiency.metrics import DetectionResult

# 1. Colete métricas de inferência
collector = InferenceMetricsCollector(window_size=100)

# Para cada frame processado pelo modelo YOLO:
collector.add(DetectionResult(
    frame_id="frame_000001",
    video_id="550e8400-e29b-41d4-a716-446655440000",  # UUID da API
    confidence_scores=[0.91, 0.78, 0.85],
    true_positives=3,
    false_positives=0,
    false_negatives=1,
    inference_time_ms=145.2,
))

# 2. Avalie suficiência
evaluator = TrainingSufficiencyEvaluator()
result = evaluator.evaluate(collector)

# 3. Gere o relatório
report = SufficiencyReport(result, model_name="yolov8-residuos-v1.2")
report.print()

print(result.is_sufficient())      # True / False
print(result.needs_retraining())   # True / False
print(result.score)                # 0.0 – 1.0
```

---

## Detecção de Deriva (Data Drift)

```python
from mlops_sufficiency import DataDriftDetector

detector = DataDriftDetector(
    confidence_drop_threshold=0.05,   # queda de 5% na confiança
    f1_drop_threshold=0.05,
    fp_rate_increase_threshold=0.05,
)

# Define o baseline após o treinamento inicial
detector.set_baseline_from_collector(collector_inicial)

# Em produção, avalia periodicamente
drift_report = detector.evaluate(collector_producao)

print(drift_report.severity)                # none / low / medium / high
print(drift_report.is_retraining_needed())  # True / False
print(drift_report.recommendation)
```

---

## Integração com a API de Vídeos

```python
from mlops_sufficiency import VideoAPIClient

client = VideoAPIClient(
    base_url="http://localhost:3000",
    token="SEU_JWT_TOKEN",
)

# Lista vídeos processados
videos = client.list_processed_videos()

# Estatísticas do pipeline
stats = client.pipeline_stats()
print(stats)  # {"Enviado": 10, "Processado": 47, "Erro ao processar": 2, "total": 59}
```

---

## Critérios de Suficiência (configuráveis)

```python
from mlops_sufficiency.evaluator import SufficiencyThresholds

thresholds = SufficiencyThresholds(
    min_mean_confidence=0.70,        # confiança mínima média
    min_mean_precision=0.75,         # precisão mínima
    min_mean_recall=0.70,            # recall mínimo
    min_mean_f1=0.72,                # F1 mínimo
    max_false_positive_rate=0.15,    # máximo de FP
    max_false_negative_rate=0.20,    # máximo de FN
    max_inference_time_ms=500.0,     # latência máxima (ms)
    max_confidence_std=0.12,         # volatilidade máxima
    min_samples=200,                 # amostras mínimas para avaliar
)
```

---

## Executar Testes

```bash
python -m pytest tests/ -v
```

---

## Status de Saída

| Status | Significado |
|---|---|
| `sufficient` | Modelo pronto. Pode ir para produção. |
| `needs_more_data` | Menos de `min_samples` amostras coletadas. |
| `needs_retraining` | Métricas abaixo dos limiares. Re-treinar. |
| `degraded` | Em produção, mas com deriva severa. Re-treinar urgente. |

---

## Integração com o Pipeline MLOps

```
Vídeo ──► /movies/import ──► YOLO inference ──► InferenceMetricsCollector
                                                          │
                                          TrainingSufficiencyEvaluator
                                                          │
                                    ┌─────────────────────┴──────────────────┐
                                    │                                        │
                              sufficient                             needs_retraining
                                    │                                        │
                             Deploy produção                     Trigger re-train pipeline
```

---

## Estrutura do Projeto

```
mlops_sufficiency/
├── mlops_sufficiency/
│   ├── __init__.py
│   ├── metrics.py          # InferenceMetricsCollector, DetectionResult
│   ├── drift.py            # DataDriftDetector, DriftReport
│   ├── evaluator.py        # TrainingSufficiencyEvaluator, SufficiencyResult
│   ├── report.py           # SufficiencyReport
│   └── api_client.py       # VideoAPIClient
├── tests/
│   └── test_mlops_sufficiency.py
├── example_uso_completo.py
├── setup.py
└── README.md
```

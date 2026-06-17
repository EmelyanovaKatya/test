# LLM Distillation: Phi-3 → TinyLlama на SQuAD

Дистилляция знаний от дообученной decoder-LLM (teacher) к компактной модели (student) на задаче Extractive Question Answering. Пайплайн: **baseline → QLoRA fine-tuning → knowledge distillation → сравнение со специализированной моделью**.

## Задача

**Extractive QA**: на входе контекст (статья Wikipedia) и вопрос, на выходе — точный span текста, отвечающий на вопрос. Это не генерация, а поиск границ ответа в исходном тексте. Промпт фиксирован для всех стадий пайплайна.

**Датасет**: [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) (~87k примеров). В проекте используется 2048 примеров для обучения и 256 для теста (с фиксированным shuffle, 42 уникальных топика в тестовой выборке).

**Метрики**: Exact Match (EM) и F1 — стандартные метрики SQuAD.

## Датасет: SQuAD подробнее

[SQuAD (Stanford Question Answering Dataset)](https://huggingface.co/datasets/rajpurkar/squad) — академический бенчмарк для задачи reading comprehension, созданный в Stanford. Содержит более 100 000 пар вопрос-ответ по 500+ статьям Wikipedia. Вопросы составлены людьми (crowdworkers) на основе реальных абзацев текста, а ответ на каждый вопрос — это непрерывный фрагмент (span) того же абзаца, без неотвечаемых вопросов (в отличие от SQuAD 2.0).

### Структура одного примера

Каждый элемент датасета — словарь с пятью полями:

```json
{
  "id": "5733be284776f41900661182",
  "title": "University_of_Notre_Dame",
  "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart...",
  "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
  "answers": {
    "text": ["Saint Bernadette Soubirous"],
    "answer_start": [515]
  }
}
```

| Поле | Тип | Назначение |
|---|---|---|
| `id` | строка | Уникальный идентификатор вопроса |
| `title` | строка | Название статьи Wikipedia (используется в проекте для анализа по топикам) |
| `context` | строка | Абзац текста, в котором содержится ответ |
| `question` | строка | Вопрос, заданный к этому абзацу |
| `answers.text` | список строк | Один или несколько вариантов правильного ответа (в test/validation split часто несколько — размечены разными аннотаторами) |
| `answers.answer_start` | список чисел | Позиция начала ответа в `context` (индекс символа) |

### Примеры вопрос-ответ

**Пример 1** — вопрос о месте действия:
- Context: *«Tesla was renowned for his work as the father of the alternating current (AC) electrical supply system. He designed the modern alternating current (AC) electricity supply system, used in most of the world today, while working at his lab in Colorado Springs.»*
- Question: *«Where was Tesla's lab located when he designed the modern AC electricity system?»*
- Answer: **«Colorado Springs»**

**Пример 2** — вопрос с числовым значением:
- Context: *«Beyoncé Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child...»*
- Question: *«In what city and state did Beyoncé grow up?»*
- Answer: **«Houston, Texas»**

**Пример 3** — несколько gold-ответов (характерно для validation split):
- Context: *«Precipitation is any product of the condensation of atmospheric water vapor that falls under gravity. The main forms of precipitation include drizzle, rain, sleet, snow, graupel and hail.»*
- Question: *«What falls under gravity after condensation of water vapor?»*
- Answers: **["Precipitation", "precipitation", "Precipitation is any product..."]** — разные аннотаторы дали разную длину/регистр ответа на один и тот же вопрос; при подсчёте метрик берётся максимум по всем вариантам

### Особенности, важные для проекта

- **Ответ — это всегда span**, а не сгенерированный текст: модель должна процитировать точный фрагмент контекста, а не сформулировать ответ своими словами. Поэтому промпт в проекте явно требует «короткий точный span из контекста».
- **Несколько gold-ответов на test/validation** — при расчёте EM и F1 для каждого примера берётся лучший результат среди всех вариантов ответа.
- **Поле `title`** используется в проекте для анализа качества student-модели (P3) по топикам — выборка из 256 тестовых примеров охватывает 42 уникальные темы Wikipedia.
- **Артикли и пунктуация игнорируются** при нормализации перед сравнением (`a`, `an`, `the` удаляются, текст приводится к lowercase) — иначе модель получала бы 0 баллов за стилистически верный, но не побуквенно идентичный ответ.

## Архитектура пайплайна

| Стадия | Описание |
|---|---|
| **P1** | Базовый teacher (Phi-3) — inference без дообучения |
| **P2** | Teacher дообучен через QLoRA (1 эпоха) |
| **P3** | Student (TinyLlama) обучен дистилляцией от teacher (KL + CE лосс, early stopping) |
| **P4** | Готовая SQuAD-модель (RoBERTa) — верхняя планка качества для сравнения |

Teacher для дистилляции (P3) выбирается автоматически: если ΔF1(P2−P1) ≥ 0.01, используется QLoRA-версия, иначе — базовая.

## Модели

**Teacher — microsoft/Phi-3-mini-4k-instruct**
- 3.8B параметров, decoder-only (RoPE + SwiGLU + RMSNorm)
- Instruct-версия, следует промптам
- Vocab size: 32 064, контекст: 4096 токенов

**Student — TinyLlama/TinyLlama-1.1B-Chat-v1.0**
- 1.1B параметров (в 3.5× меньше teacher)
- LLaMA-совместимая архитектура, 22 слоя, hidden size 2048
- Chat-версия (SFT + RLHF), vocab size: 32 000

## Технические решения

**4-bit квантизация (bitsandbytes)** — веса в NF4, double quantization констант, вычисления в fp16. Экономия памяти ~4×, что позволяет держать teacher и student одновременно на GPU с `max_memory={0: "10GiB"}`.

**QLoRA / LoRA** — обучаются только низкоранговые адаптеры (`W' = W + B·A`, rank r=16) на проекциях `q_proj, k_proj, v_proj, o_proj`. Обучаемых параметров — около 1% от полной модели.

**Дистилляция знаний** — итоговый лосс студента:

```
L = α · CE(student_logits, labels) + β · KL(student || teacher)
α = 0.5, β = 0.5, температура T = 2.0
```

Поскольку у teacher и student разные словари (32 064 vs 32 000), KL-дивергенция считается только по общей части (`min_vocab = 32000`), логиты обрезаются, а labels вне диапазона маскируются через `-100`.

**Контроль переобучения** — 15% train выделено в val split (~307 примеров), CosineAnnealingLR, gradient clipping (max_norm=1.0), early stopping (patience=2 эпохи по val loss).

## Результаты

| Stage | EM | F1 | Throughput (samples/sec) |
|---|---:|---:|---:|
| P1 — baseline teacher | 14.45% | 43.09% | 0.34 |
| P2 — QLoRA teacher | 37.11% | 58.58% | 0.44 |
| P3 — distilled student | 55.47% | 67.80% | 1.03 |
| P4 — SQuAD-pretrained (RoBERTa) | 85.55% | 89.57% | 59.33 |

- QLoRA дообучение teacher (1 эпоха, ~2048 примеров, T4) поднимает EM с 14.4% до 37.1%.
- Дистилляция от дообученного teacher даёт student прирост в +18 EM относительно P2 — student перенимает не только правильные токены, но и формат ответа (короткий span, а не многословное объяснение).
- Student (1.1B) работает в 2.3× быстрее teacher (3.8B) при ощутимо более высоком EM/F1 на тесте.
- P4 (специализированная encoder-модель, обученная именно под SQuAD) показывает верхнюю границу качества для задачи — наглядно, насколько decoder-пайплайн уступает узкоспециализированному решению.

## Технические сложности и решения

| Проблема | Решение |
|---|---|
| OOM при одновременной загрузке teacher + student | Обе модели в 4-bit, явный лимит `max_memory={0: "10GiB"}` |
| Разные словари (32 064 vs 32 000) | Обрезка логитов до `min_vocab`, маскирование лишних labels через `-100` |
| NaN в лоссе (fp16 overflow при KL) | Каст в float32 перед softmax/log_softmax |
| Высокая дисперсия метрик на малой тестовой выборке | Shuffle + увеличение теста до 256 примеров (снижает разброс EM с ±3.1% до ±0.4%) |

## Ограничения

- Обучение teacher — всего 1 эпоха на 2048 примерах.
- Дистилляция работает на token-level KL, без sequence-level дистилляции (нет учёта целостных сгенерированных последовательностей).
- Объём train/test ограничен ресурсами GPU (T4/L4 на Kaggle).

## Инфраструктура

- Платформа: Kaggle (GPU T4×2 или L4), совместимо с Google Colab и локальным запуском.
- Стек: `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`, `evaluate`, `accelerate`.
- Воспроизводимость: фиксированный `GLOBAL_SEED=42` для всех генераторов случайных чисел.
- Промежуточные результаты каждой стадии кэшируются в JSON — повторный запуск не пересчитывает уже готовые стадии (если не установлен `FORCE_RERUN_STAGES=1`).

## Структура артефактов

```
artifacts/
├── p1_baseline_metrics.json
├── p2_qlora_metrics.json
├── teacher_qlora_adapter/        # сохранённые LoRA-веса teacher
├── teacher_choice.json           # обоснование выбора teacher для P3
├── p3_student_metrics.json
├── student_distilled/            # сохранённые LoRA-веса student
├── p4_squad_teacher_metrics.json
├── distill_loss_curves.png       # train/val loss по эпохам
├── graph1_em_f1.png              # сравнение EM/F1 по стадиям
├── final_report.md
└── run_manifest.json             # метаданные запуска (seed, GPU, версии)
```

## Запуск

Ноутбук автоматически определяет среду выполнения (Kaggle / Colab / локально) и устанавливает недостающие зависимости. Для полного перезапуска всех стадий с нуля:

```bash
export FORCE_RERUN_STAGES=1
```

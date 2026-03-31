#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# run_full_pipeline.sh — Полный пайплайн ProcessScout AI для БП 06.01-06.14
#
# Каждый шаг сохраняет результат в файл. Если скрипт упадёт — перезапусти,
# он пропустит уже готовые файлы (проверка через [ -f ... ]).
#
# Использование:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh 2>&1 | tee output/full_pipeline.log
# ═══════════════════════════════════════════════════════════════════════

set -e  # Остановиться при ошибке

BANK="bank_4-5.xlsx"

echo "════════════════════════════════════════════════════════════════"
echo "ProcessScout AI — Полный пайплайн"
echo "Старт: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# ФАЗА 1: МАППИНГ (cluster_search) — нужен VM + LLM
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "══════ ФАЗА 1: МАППИНГ ══════"

for BP in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
  OUTFILE="output/bp_06_${BP}/mapping/results_clustered.json"
  
  if [ -f "$OUTFILE" ]; then
    echo "[SKIP] 06.${BP} маппинг — уже есть: ${OUTFILE}"
    continue
  fi
  
  echo ""
  echo "═══ 06.${BP} маппинг started at $(date) ═══"
  python3 scripts/cluster_search.py \
    --asis "output/bp_06_${BP}/chunks/chunks_3types.json" \
    --top_clusters 5 \
    --output_dir "output/bp_06_${BP}/mapping"
  echo "═══ 06.${BP} маппинг done at $(date) ═══"
done

echo ""
echo "══════ ФАЗА 1 ЗАВЕРШЕНА: все маппинги готовы ══════"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# ФАЗА 2: КОНСОЛИДАЦИЯ ШАГОВ (без LLM, быстро)
# ═══════════════════════════════════════════════════════════════════════

echo "══════ ФАЗА 2: КОНСОЛИДАЦИЯ ШАГОВ ══════"

for BP in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
  OUTFILE="output/bp_06_${BP}/mapping/consolidated_l3.json"
  
  if [ -f "$OUTFILE" ]; then
    echo "[SKIP] 06.${BP} consolidated_l3 — уже есть"
    continue
  fi
  
  echo "[RUN] 06.${BP} consolidate_results..."
  python3 scripts/consolidate_results.py \
    --input "output/bp_06_${BP}/mapping/results_clustered.json" \
    --output "$OUTFILE"
done

echo ""
echo "══════ ФАЗА 2 ЗАВЕРШЕНА ══════"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# ФАЗА 3: КОНСОЛИДАЦИЯ РЕКОМЕНДАЦИЙ (без LLM, быстро)
# ═══════════════════════════════════════════════════════════════════════

echo "══════ ФАЗА 3: КОНСОЛИДАЦИЯ РЕКОМЕНДАЦИЙ ══════"

for BP in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
  OUTFILE="output/bp_06_${BP}/mapping/consolidated_recs.json"
  INFILE="output/bp_06_${BP}/mapping/results_clustered.json"
  
  if [ -f "$OUTFILE" ]; then
    echo "[SKIP] 06.${BP} consolidated_recs — уже есть"
    continue
  fi
  
  echo "[RUN] 06.${BP} consolidate_recs..."
  python3 scripts/consolidate_recs.py \
    --input "$INFILE" \
    --output "$OUTFILE"
done

# Особый случай: 06.01 рекомендации прогнаны отдельно
# Если consolidated_recs.json пустой или отсутствует, используем mapping_recs
if [ -f "output/bp_06_01/mapping_recs/results_clustered.json" ]; then
  RECS01="output/bp_06_01/mapping/consolidated_recs.json"
  if [ ! -f "$RECS01" ] || [ "$(python3 -c "import json; d=json.load(open('$RECS01')); print(len(d.get('consolidated',[])))" 2>/dev/null)" = "0" ]; then
    echo "[FIX] 06.01 рекомендации из mapping_recs..."
    python3 scripts/consolidate_recs.py \
      --input "output/bp_06_01/mapping_recs/results_clustered.json" \
      --output "$RECS01"
  fi
fi

echo ""
echo "══════ ФАЗА 3 ЗАВЕРШЕНА ══════"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# ФАЗА 4: РАСКРЫТИЕ ДЛЯ ШАГОВ — нужен VM + LLM
# ═══════════════════════════════════════════════════════════════════════

echo "══════ ФАЗА 4: РАСКРЫТИЕ ШАГОВ ══════"

for BP in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
  DIR="output/bp_06_${BP}"
  CONSOL="${DIR}/mapping/consolidated_l3.json"
  RESULTS="${DIR}/mapping/results_clustered.json"
  D1="${DIR}/expansion/expansion_d1.json"
  D2="${DIR}/expansion/expansion_d2.json"
  EC="${DIR}/expansion/expansion_consolidated.json"
  
  # d1
  if [ -f "$D1" ]; then
    echo "[SKIP] 06.${BP} steps d1 — уже есть"
  else
    echo "[RUN] 06.${BP} steps d1..."
    python3 scripts/expand_direction1_cluster.py \
      --consolidated "$CONSOL" \
      --results "$RESULTS" \
      --bank "$BANK" \
      --output "$D1"
  fi
  
  # d2
  if [ -f "$D2" ]; then
    echo "[SKIP] 06.${BP} steps d2 — уже есть"
  else
    echo "[RUN] 06.${BP} steps d2..."
    python3 scripts/expand_direction2_hierarchy.py \
      --consolidated "$CONSOL" \
      --bank "$BANK" \
      --output "$D2"
  fi
  
  # consolidate expansion
  if [ -f "$EC" ]; then
    echo "[SKIP] 06.${BP} steps expansion_consolidated — уже есть"
  else
    echo "[RUN] 06.${BP} steps consolidate_expansion..."
    python3 scripts/consolidate_expansion.py \
      --d1 "$D1" \
      --d2 "$D2" \
      --output "$EC"
  fi
done

echo ""
echo "══════ ФАЗА 4 ЗАВЕРШЕНА ══════"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# ФАЗА 5: РАСКРЫТИЕ ДЛЯ РЕКОМЕНДАЦИЙ — нужен VM + LLM
# ═══════════════════════════════════════════════════════════════════════

echo "══════ ФАЗА 5: РАСКРЫТИЕ РЕКОМЕНДАЦИЙ ══════"

for BP in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
  DIR="output/bp_06_${BP}"
  CONSOL_RECS="${DIR}/mapping/consolidated_recs.json"
  
  # Пропускаем если нет consolidated_recs (нет рекомендаций)
  if [ ! -f "$CONSOL_RECS" ]; then
    echo "[SKIP] 06.${BP} recs — нет consolidated_recs.json"
    continue
  fi
  
  # Проверяем что рекомендации не пустые
  NRECS=$(python3 -c "import json; d=json.load(open('$CONSOL_RECS')); print(len(d.get('consolidated',[])))" 2>/dev/null || echo "0")
  if [ "$NRECS" = "0" ]; then
    echo "[SKIP] 06.${BP} recs — 0 рекомендаций"
    continue
  fi
  
  # Определяем файл results для d1
  # Для 06.01 рекомендации в отдельном файле
  if [ "$BP" = "01" ] && [ -f "${DIR}/mapping_recs/results_clustered.json" ]; then
    RESULTS_RECS="${DIR}/mapping_recs/results_clustered.json"
  else
    RESULTS_RECS="${DIR}/mapping/results_clustered.json"
  fi
  
  RD1="${DIR}/expansion/expansion_recs_d1.json"
  RD2="${DIR}/expansion/expansion_recs_d2.json"
  REC="${DIR}/expansion/expansion_recs_consolidated.json"
  
  # d1 для рекомендаций (может дать 0 соседей — это нормально)
  if [ -f "$RD1" ]; then
    echo "[SKIP] 06.${BP} recs d1 — уже есть"
  else
    echo "[RUN] 06.${BP} recs d1 (${NRECS} рекомендаций)..."
    python3 scripts/expand_direction1_cluster.py \
      --consolidated "$CONSOL_RECS" \
      --results "$RESULTS_RECS" \
      --bank "$BANK" \
      --output "$RD1" || {
        echo "[WARN] 06.${BP} recs d1 failed, создаю пустой файл"
        echo '{"expansions":[]}' > "$RD1"
      }
  fi
  
  # d2 для рекомендаций
  if [ -f "$RD2" ]; then
    echo "[SKIP] 06.${BP} recs d2 — уже есть"
  else
    echo "[RUN] 06.${BP} recs d2..."
    python3 scripts/expand_direction2_hierarchy.py \
      --consolidated "$CONSOL_RECS" \
      --bank "$BANK" \
      --output "$RD2"
  fi
  
  # consolidate expansion рекомендаций
  if [ -f "$REC" ]; then
    echo "[SKIP] 06.${BP} recs expansion_consolidated — уже есть"
  else
    echo "[RUN] 06.${BP} recs consolidate_expansion..."
    python3 scripts/consolidate_expansion.py \
      --d1 "$RD1" \
      --d2 "$RD2" \
      --output "$REC"
  fi
done

echo ""
echo "══════ ФАЗА 5 ЗАВЕРШЕНА ══════"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# ФИНАЛ: ПРОВЕРКА
# ═══════════════════════════════════════════════════════════════════════

echo "══════ ФИНАЛЬНАЯ ПРОВЕРКА ══════"
echo ""

for BP in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
  DIR="output/bp_06_${BP}"
  
  # Считаем файлы
  MAPPING=$([ -f "${DIR}/mapping/results_clustered.json" ] && echo "✅" || echo "❌")
  CONSOL=$([ -f "${DIR}/mapping/consolidated_l3.json" ] && echo "✅" || echo "❌")
  RECS=$([ -f "${DIR}/mapping/consolidated_recs.json" ] && echo "✅" || echo "—")
  EXP=$([ -f "${DIR}/expansion/expansion_consolidated.json" ] && echo "✅" || echo "❌")
  REXP=$([ -f "${DIR}/expansion/expansion_recs_consolidated.json" ] && echo "✅" || echo "—")
  
  echo "06.${BP}: mapping=${MAPPING} consol=${CONSOL} recs=${RECS} exp_steps=${EXP} exp_recs=${REXP}"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Финиш: $(date)"
echo "════════════════════════════════════════════════════════════════"
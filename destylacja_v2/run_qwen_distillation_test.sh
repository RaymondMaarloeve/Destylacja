#!/bin/bash
# TESTOWA wersja destylacji - tylko 20 promptów dla szybkiego testu
# Sprawdza czy cały pipeline działa poprawnie

set -e  # Przerwij przy błędzie

echo "════════════════════════════════════════════════════════════"
echo "  🧪 TEST PIPELINE DESTYLACJI (20 promptów)"
echo "════════════════════════════════════════════════════════════"
echo ""

# Konfiguracja testowa
TEST_DATASET="/root/destylacja/dataset_test.json"
RESPONSES_FILE="/root/destylacja/qwen_responses_test.json"
MODEL_DIR="/root/destylacja/models/qwen-3b-distilled_test"

# Stwórz testowy dataset z pierwszych 20 przykładów
echo "📝 Tworzenie testowego datasetu (20 promptów)..."
python3 -c "
import json
with open('/root/destylacja/dataset.json', 'r') as f:
    data = json.load(f)
with open('${TEST_DATASET}', 'w') as f:
    json.dump(data[:20], f, ensure_ascii=False, indent=2)
print('✓ Testowy dataset stworzony: ${TEST_DATASET}')
"

echo ""
echo "Konfiguracja testowa:"
echo "  - Nauczyciel: Qwen2.5-7B-Instruct"
echo "  - Student: Qwen2.5-3B-Instruct"
echo "  - Dataset: 20 promptów (test)"
echo "  - Batch size: 4 (mniejszy dla testu)"
echo "  - Max tokens: 256"
echo "  - Epoki treningu: 1 (zamiast 3)"
echo ""
echo "Szacowany czas: ~15-20 minut"
echo "  - Faza 1 (generowanie): ~2-3 minuty"
echo "  - Faza 2 (trening): ~10-15 minut"
echo ""
read -p "Kontynuować test? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Anulowano."
    exit 1
fi

# ════════════════════════════════════════════════════════════
# FAZA 1: Generowanie odpowiedzi Qwen 7B
# ════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  FAZA 1: Generowanie odpowiedzi (Qwen2.5-7B-Instruct)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⏰ Start Fazy 1: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /root/destylacja

python3 generate_qwen_responses.py \
  --input "${TEST_DATASET}" \
  --output "${RESPONSES_FILE}" \
  --batch-size 4 \
  --max-tokens 256

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ BŁĄD: Faza 1 nie powiodła się!"
    exit 1
fi

echo ""
echo "✓ Faza 1 zakończona pomyślnie!"
echo "⏰ Koniec Fazy 1: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Sprawdź czy plik istnieje
if [ ! -f "${RESPONSES_FILE}" ]; then
    echo "✗ BŁĄD: Nie znaleziono pliku ${RESPONSES_FILE}"
    exit 1
fi

RESPONSE_COUNT=$(python3 -c "import json; print(len(json.load(open('${RESPONSES_FILE}'))))")
echo "📊 Wygenerowano ${RESPONSE_COUNT} odpowiedzi"

# Wyczyść cache GPU przed Fazą 2
echo ""
echo "🧹 Czyszczenie cache GPU..."
python3 -c "import torch; torch.cuda.empty_cache(); print('✓ GPU cache wyczyszczony')"
sleep 5

# ════════════════════════════════════════════════════════════
# FAZA 2: Trening studenta (Qwen 3B) - TESTOWY
# ════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  FAZA 2: Fine-tuning studenta (Qwen2.5-3B-Instruct)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⏰ Start Fazy 2: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /root/destylacja/src

python3 train_student.py \
  --teacher-responses "${RESPONSES_FILE}" \
  --student-model Qwen/Qwen2.5-3B-Instruct \
  --output-dir "${MODEL_DIR}" \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --learning-rate 5e-5

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ BŁĄD: Faza 2 nie powiodła się!"
    exit 1
fi

echo ""
echo "✓ Faza 2 zakończona pomyślnie!"
echo "⏰ Koniec Fazy 2: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ════════════════════════════════════════════════════════════
# ZAKOŃCZENIE TESTU
# ════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✨ TEST PIPELINE ZAKOŃCZONY POMYŚLNIE! ✨"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⏰ Całkowity czas zakończenia: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📁 Pliki testowe:"
echo "  • Testowy dataset:"
echo "    ${TEST_DATASET}"
echo ""
echo "  • Odpowiedzi nauczyciela:"
echo "    ${RESPONSES_FILE}"
echo ""
echo "  • Wytrenowany model testowy:"
echo "    ${MODEL_DIR}"
echo ""
echo "🧪 Testowanie modelu:"
echo "  cd /root/destylacja/src"
echo "  python3 test_model.py --model-path ${MODEL_DIR} --interactive"
echo ""
echo "💡 Jeśli test działa poprawnie, uruchom pełną destylację:"
echo "  ./run_qwen_distillation.sh"
echo ""
echo "🧹 Aby usunąć pliki testowe:"
echo "  rm ${TEST_DATASET} ${RESPONSES_FILE}"
echo "  rm -rf ${MODEL_DIR}"
echo ""
echo "════════════════════════════════════════════════════════════"

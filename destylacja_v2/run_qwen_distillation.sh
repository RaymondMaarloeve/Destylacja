#!/bin/bash
# Automatyczna destylacja Qwen2.5-7B-Instruct → Qwen2.5-3B-Instruct
# Faza 1: Generowanie odpowiedzi przez Qwen 7B
# Faza 2: Trening Qwen 3B na odpowiedziach

set -e  # Przerwij przy błędzie

# Konfiguracja
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESPONSES_FILE="/root/destylacja/qwen_responses_${TIMESTAMP}.json"
MODEL_DIR="/root/destylacja/models/qwen-3b-distilled_${TIMESTAMP}"

# Backup poprzednich plików jeśli istnieją
if [ -f "/root/destylacja/qwen_responses.json" ]; then
    cp /root/destylacja/qwen_responses.json /root/destylacja/qwen_responses_backup.json
    echo "✓ Backup: qwen_responses.json → qwen_responses_backup.json"
fi

echo "════════════════════════════════════════════════════════════"
echo "  AUTOMATYCZNA DESTYLACJA QWEN"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Konfiguracja:"
echo "  - Nauczyciel: Qwen2.5-7B-Instruct"
echo "  - Student: Qwen2.5-3B-Instruct"
echo "  - Dataset: $(jq length /root/destylacja/dataset.json) promptów"
echo "  - Batch size: 8"
echo "  - Max tokens: 512"
echo "  - Epoki treningu: 3"
echo ""
echo "Pliki wyjściowe:"
echo "  - Odpowiedzi: ${RESPONSES_FILE}"
echo "  - Model: ${MODEL_DIR}"
echo ""
echo "Szacowany czas: ~8-10 godzin"
echo "  - Faza 1 (generowanie): ~4 godziny"
echo "  - Faza 2 (trening): ~4-6 godzin"
echo ""
read -p "Kontynuować? (y/n) " -n 1 -r
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
echo "⏰ Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /root/destylacja

python3 generate_qwen_responses.py \
  --input /root/destylacja/dataset.json \
  --output "${RESPONSES_FILE}" \
  --batch-size 8 \
  --max-tokens 256

# Utwórz link do najnowszej wersji
ln -sf "$(basename ${RESPONSES_FILE})" /root/destylacja/qwen_responses_latest.json

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ BŁĄD: Faza 1 nie powiodła się!"
    exit 1
fi

echo ""
echo "✓ Faza 1 zakończona pomyślnie!"
echo "⏰ Koniec Fazy 1: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Sprawdź czy plik istnieje i ma poprawną strukturę
if [ ! -f "${RESPONSES_FILE}" ]; then
    echo "✗ BŁĄD: Nie znaleziono pliku ${RESPONSES_FILE}"
    exit 1
fi

RESPONSE_COUNT=$(jq length "${RESPONSES_FILE}")
echo "📊 Wygenerowano ${RESPONSE_COUNT} odpowiedzi"

# Wyczyść cache GPU przed Fazą 2
echo ""
echo "🧹 Czyszczenie cache GPU..."
python3 -c "import torch; torch.cuda.empty_cache(); print('✓ GPU cache wyczyszczony')"
sleep 5

# ════════════════════════════════════════════════════════════
# FAZA 2: Trening studenta (Qwen 3B)
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
  --epochs 3 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --learning-rate 5e-5

# Utwórz link do najnowszego modelu
cd /root/destylacja/models
ln -sf "$(basename ${MODEL_DIR})" qwen-3b-distilled_latest

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
# ZAKOŃCZENIE
# ════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✨ DESTYLACJA ZAKOŃCZONA POMYŚLNIE! ✨"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⏰ Całkowity czas zakończenia: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📁 Pliki wyjściowe:"
echo "  • Odpowiedzi nauczyciela:"
echo "    ${RESPONSES_FILE}"
echo ""
echo "  • Wytrenowany model:"
echo "    ${MODEL_DIR}"
echo ""
echo "  • Link do najnowszego:"
echo "    /root/destylacja/models/qwen-3b-distilled_latest/"
echo ""
echo "🧪 Testowanie modelu:"
echo "  cd /root/destylacja/src"
echo "  python3 test_model.py --model-path ../models/qwen-3b-distilled_latest --interactive"
echo ""
echo "Lub pojedynczy prompt:"
echo "  python3 test_model.py --model-path ../models/qwen-3b-distilled_latest --prompt 'Hello!'"
echo ""
echo "════════════════════════════════════════════════════════════"

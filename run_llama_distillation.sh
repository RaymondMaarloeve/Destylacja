#!/bin/bash

# DESTYLACJA LLAMA 11B → 3B
# Pełny pipeline: generacja danych + trening + ewaluacja

echo "=========================================="
echo "🚀 Destylacja Llama-3.2-11B → 3B"
echo "=========================================="
echo ""
echo "Nauczyciel: Llama-3.2-11B-Vision-Instruct"
echo "Student: Llama-3.2-3B-Instruct"
echo "Samples: 2878 (wszystkie z Excel)"
echo "Czas: ~4-8 godzin"
echo ""

# Kolory
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Sprawdź venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment nie jest aktywowane!${NC}"
    echo "Uruchom: source venv/bin/activate"
    exit 1
fi

# Sprawdź GPU
echo -e "${BLUE}Sprawdzam GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Sprawdź Excel
if [ ! -f "prompty.xlsx" ]; then
    echo -e "${RED}❌ Brak pliku prompty.xlsx!${NC}"
    exit 1
fi

read -p "Kontynuować? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Anulowano."
    exit 0
fi

echo ""
echo -e "${BLUE}[1/3] Generowanie danych od nauczyciela (2-4h)...${NC}"
echo "Llama-11B będzie generować odpowiedzi na 2878 promptów z Excel"
echo ""

python scripts/generate_teacher_data.py --config config_llama3_70b_3b.yaml

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Błąd podczas generowania danych!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dane wygenerowane${NC}\n"

echo -e "${BLUE}[2/3] Trening destylacji (2-4h)...${NC}"
echo "Student Llama-3B uczy się od nauczyciela Llama-11B"
echo ""

python scripts/train_distillation.py --config config_llama3_70b_3b.yaml

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Błąd podczas treningu!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Destylacja zakończona${NC}\n"

echo -e "${BLUE}[3/3] Ewaluacja modelu...${NC}"
python scripts/evaluate_model.py \
    --student_path models/llama3_11b_to_3b_npc \
    --config config_llama3_70b_3b.yaml \
    --teacher_name meta-llama/Llama-3.2-11B-Vision-Instruct \
    --mode compare

echo ""
echo "=========================================="
echo -e "${GREEN}🏆 DESTYLACJA ZAKOŃCZONA! 🏆${NC}"
echo "=========================================="
echo ""
echo "📁 Pliki:"
echo "   Model: models/llama3_11b_to_3b_npc/"
echo "   Dataset: data/teacher_dataset.jsonl"
echo ""
echo "🎮 Następne kroki:"
echo "  1. Test interaktywny:"
echo "     python scripts/evaluate_model.py \\"
echo "       --student_path models/llama3_11b_to_3b_npc \\"
echo "       --config config_llama3_70b_3b.yaml \\"
echo "       --mode interactive"
echo ""
echo "  2. Spakuj model:"
echo "     tar -czf llama3_npc_model.tar.gz models/llama3_11b_to_3b_npc/"
echo ""

import pandas as pd
import json

# Wczytaj wszystkie arkusze z Excela
excel_file = pd.ExcelFile('../prompty(2).xlsx')
print(f"Znaleziono {len(excel_file.sheet_names)} arkuszy:")
for name in excel_file.sheet_names:
    print(f"  - {name}")

all_prompts = []

# Przejdź przez wszystkie arkusze
for sheet_name in excel_file.sheet_names:
    df = pd.read_excel('../prompty(2).xlsx', sheet_name=sheet_name)
    print(f"\nArkusz '{sheet_name}': {len(df)} wierszy")
    
    # Pobierz pierwszą kolumnę (input)
    if len(df.columns) > 0:
        first_column = df.iloc[:, 0]
        
        # Dodaj wszystkie niepuste wartości
        for value in first_column:
            if pd.notna(value) and str(value).strip():
                all_prompts.append({
                    "prompt": str(value).strip(),
                    "sheet": sheet_name
                })

print(f"\n{'='*50}")
print(f"Łącznie zebranych promptów: {len(all_prompts)}")

# Zapisz jako JSON
with open('../dataset.json', 'w', encoding='utf-8') as f:
    json.dump(all_prompts, f, ensure_ascii=False, indent=2)

# Zapisz jako JSONL
with open('../dataset.jsonl', 'w', encoding='utf-8') as f:
    for entry in all_prompts:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"✓ Zapisano do dataset.json i dataset.jsonl")
print(f"\nPrzykładowe prompty:")
for i, prompt in enumerate(all_prompts[:3]):
    print(f"  {i+1}. [{prompt['sheet']}] {prompt['prompt'][:80]}...")

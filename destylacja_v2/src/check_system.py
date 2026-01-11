#!/usr/bin/env python3
"""
Sprawdza system i dostępne zasoby dla destylacji modeli
"""
import sys
import json

def check_cuda():
    """Sprawdza CUDA i GPU"""
    try:
        import torch
        print(f"✓ PyTorch zainstalowany: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA dostępna: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"✓ Znaleziono GPU: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name}")
                print(f"    VRAM: {vram_gb:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
                
            # Sprawdź dostępny VRAM
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available = total - reserved
            
            print(f"\n  Pamięć GPU 0:")
            print(f"    Używane: {allocated:.2f} GB")
            print(f"    Zarezerwowane: {reserved:.2f} GB")
            print(f"    Dostępne: {available:.2f} GB")
            
            return True, total
        else:
            print("✗ CUDA niedostępna - brak GPU lub sterowników")
            return False, 0
    except ImportError:
        print("✗ PyTorch nie zainstalowany")
        return False, 0

def check_libraries():
    """Sprawdza wymagane biblioteki"""
    libs = {
        'transformers': 'transformers',
        'bitsandbytes': 'bitsandbytes',
        'accelerate': 'accelerate',
        'peft': 'peft',
        'trl': 'trl',
        'datasets': 'datasets'
    }
    
    missing = []
    for name, import_name in libs.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: nie zainstalowany")
            missing.append(name)
    
    return len(missing) == 0, missing

def estimate_vram_requirements():
    """Szacuje wymagania VRAM"""
    print("\n" + "="*60)
    print("SZACOWANE WYMAGANIA VRAM:")
    print("="*60)
    
    configs = {
        "Llama 3.1 70B (4-bit quantized)": {
            "inference": 40,
            "description": "Nauczyciel - tylko inference"
        },
        "Llama 3.2 3B (pełna precyzja)": {
            "inference": 6,
            "training_full": 18,
            "training_lora": 8,
            "description": "Student"
        }
    }
    
    for model, req in configs.items():
        print(f"\n{model}:")
        print(f"  {req['description']}")
        for mode, vram in req.items():
            if mode != 'description':
                print(f"  {mode}: {vram} GB")
    
    print("\n" + "="*60)
    print("REKOMENDOWANE PODEJŚCIE (dla 48GB VRAM):")
    print("="*60)
    print("Faza 1: Generowanie odpowiedzi nauczyciela")
    print("  Llama 70B 4-bit: ~40 GB")
    print("  Status: ✓ Zmieści się")
    print("\nFaza 2: Trening studenta")
    print("  Llama 3B + LoRA: ~8 GB")
    print("  Status: ✓ Zmieści się")
    print("\nObie fazy jednocześnie: 40 + 8 = 48 GB")
    print("  Status: ⚠️  Na granicy - lepiej osobno")

def main():
    print("="*60)
    print("SPRAWDZANIE SYSTEMU DLA DESTYLACJI MODELI")
    print("="*60)
    print(f"Python: {sys.version}")
    
    print("\n" + "="*60)
    print("SPRAWDZANIE CUDA I GPU:")
    print("="*60)
    has_cuda, vram_total = check_cuda()
    
    print("\n" + "="*60)
    print("SPRAWDZANIE BIBLIOTEK:")
    print("="*60)
    has_libs, missing = check_libraries()
    
    estimate_vram_requirements()
    
    print("\n" + "="*60)
    print("PODSUMOWANIE:")
    print("="*60)
    
    if has_cuda and has_libs:
        print("✓ System gotowy do destylacji!")
        if vram_total >= 48:
            print(f"✓ VRAM ({vram_total:.1f} GB) wystarczający")
        else:
            print(f"⚠️  VRAM ({vram_total:.1f} GB) może być niewystarczający")
            print("   Rozważ użycie Llama 8B jako nauczyciela")
    else:
        print("✗ System wymaga konfiguracji:")
        if not has_cuda:
            print("  - Zainstaluj sterowniki CUDA")
            print("  - Zainstaluj PyTorch z CUDA")
        if not has_libs:
            print(f"  - Zainstaluj brakujące biblioteki:")
            print(f"    pip install {' '.join(missing)}")
    
    return has_cuda and has_libs

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

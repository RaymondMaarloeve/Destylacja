#!/usr/bin/env python3
"""
Kompletna diagnostyka środowiska serwera.
Uruchom to NA SERWERZE żeby znaleźć problem.
"""

import sys
import os

print("=" * 70)
print("🔍 DIAGNOSTYKA ŚRODOWISKA SERWERA")
print("=" * 70)

# 1. Python version
print(f"\n1️⃣  Python Version:")
print(f"   {sys.version}")
print(f"   Executable: {sys.executable}")

# 2. CUDA environment
print(f"\n2️⃣  CUDA Environment Variables:")
cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH']
for var in cuda_vars:
    val = os.environ.get(var, 'NOT SET')
    print(f"   {var}: {val}")

# 3. PyTorch
print(f"\n3️⃣  PyTorch:")
try:
    import torch
    print(f"   ✓ Version: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      Memory: {props.total_memory / 1024**3:.1f}GB")
    else:
        print(f"   ❌ CUDA NOT AVAILABLE!")
        print(f"   ❌ PyTorch zainstalowany BEZ wsparcia CUDA!")
        print(f"\n   FIX: Przeinstaluj PyTorch:")
        print(f"   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
except ImportError:
    print(f"   ❌ PyTorch nie zainstalowany!")

# 4. Transformers
print(f"\n4️⃣  Transformers:")
try:
    import transformers
    print(f"   ✓ Version: {transformers.__version__}")
except ImportError:
    print(f"   ❌ Transformers nie zainstalowany!")

# 5. Accelerate (używane przez device_map="auto")
print(f"\n5️⃣  Accelerate (kontroluje device_map='auto'):")
try:
    import accelerate
    print(f"   ✓ Version: {accelerate.__version__}")
except ImportError:
    print(f"   ⚠️  Accelerate nie zainstalowany!")
    print(f"   To może powodować problemy z device_map='auto'")

# 6. Bitsandbytes (dla 8-bit quantization)
print(f"\n6️⃣  Bitsandbytes (dla load_in_8bit):")
try:
    import bitsandbytes
    print(f"   ✓ Version: {bitsandbytes.__version__}")
except ImportError:
    print(f"   ⚠️  Bitsandbytes nie zainstalowany!")
    print(f"   8-bit quantization może nie działać!")

# 7. Test GPU speed
print(f"\n7️⃣  Test szybkości GPU:")
try:
    import torch
    import time
    
    if torch.cuda.is_available():
        # Small matmul test
        size = 2048
        device = "cuda:0"
        
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(3):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Measure
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        ops_per_sec = (2 * size**3 * 10) / elapsed / 1e9  # GFLOPS
        
        print(f"   ✓ Matrix multiply (2048x2048 FP16): {elapsed:.3f}s for 10 iterations")
        print(f"   ✓ Performance: {ops_per_sec:.1f} GFLOPS")
        
        if ops_per_sec < 100:
            print(f"   ⚠️  WOLNO! (powinno być >1000 GFLOPS na RTX 5000 Ada)")
        elif ops_per_sec < 1000:
            print(f"   ⚠️  Poniżej oczekiwań dla RTX 5000 Ada")
        else:
            print(f"   ✓ Szybkość OK!")
    else:
        print(f"   ❌ Brak CUDA - nie mogę przetestować GPU")
except Exception as e:
    print(f"   ❌ Błąd testu: {e}")

# 8. Test device_map="auto" behavior
print(f"\n8️⃣  Test device_map='auto':")
try:
    import torch
    from transformers import AutoModelForCausalLM
    
    if torch.cuda.is_available():
        print(f"   Ładowanie małego modelu (gpt2) z device_map='auto'...")
        
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Check where it landed
        if hasattr(model, 'hf_device_map'):
            print(f"   ✓ Device map: {model.hf_device_map}")
        
        device = next(model.parameters()).device
        print(f"   ✓ Model device: {device}")
        
        if 'cpu' in str(device):
            print(f"   ❌ PROBLEM! Model trafił na CPU mimo dostępnego GPU!")
            print(f"   device_map='auto' NIE DZIAŁA POPRAWNIE na tym serwerze!")
        else:
            print(f"   ✓ Model poprawnie na GPU")
            
        del model
        torch.cuda.empty_cache()
    else:
        print(f"   ❌ Brak CUDA")
except Exception as e:
    print(f"   ❌ Błąd: {e}")

# 9. Disk I/O speed (HuggingFace cache)
print(f"\n9️⃣  HuggingFace Cache:")
cache_dir = os.path.expanduser("~/.cache/huggingface")
print(f"   Path: {cache_dir}")
if os.path.exists(cache_dir):
    import subprocess
    try:
        # Check disk usage
        result = subprocess.run(['du', '-sh', cache_dir], capture_output=True, text=True)
        print(f"   Size: {result.stdout.strip()}")
        
        # Check if on slow filesystem
        result = subprocess.run(['df', '-h', cache_dir], capture_output=True, text=True)
        print(f"   Filesystem: {result.stdout.split()[0]}")
    except:
        pass
else:
    print(f"   ⚠️  Cache nie istnieje")

print("\n" + "=" * 70)
print("📋 PODSUMOWANIE:")
print("=" * 70)

# Summary checks
issues = []

try:
    import torch
    if not torch.cuda.is_available():
        issues.append("❌ CRITICAL: PyTorch bez CUDA")
    elif torch.cuda.device_count() < 2:
        issues.append(f"⚠️  WARNING: Wykryto {torch.cuda.device_count()} GPU (oczekiwano 2)")
except:
    issues.append("❌ CRITICAL: PyTorch nie zainstalowany")

try:
    import accelerate
except:
    issues.append("⚠️  WARNING: Brak accelerate (device_map='auto' może nie działać)")

try:
    import bitsandbytes
except:
    issues.append("⚠️  WARNING: Brak bitsandbytes (load_in_8bit nie zadziała)")

if issues:
    print("\n🔴 Znalezione problemy:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ Środowisko wygląda OK!")
    print("   Problem może być w innym miejscu (model, config, etc.)")

print("=" * 70)

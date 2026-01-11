#!/usr/bin/env python3
"""
Faza 1: Generowanie odpowiedzi nauczyciela (domyślnie Qwen2.5 7B 4-bit)
Generuje odpowiedzi dla wszystkich promptów z dataset.json
"""
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import time
from datetime import timedelta
import argparse

# Prefer szybsze matmul na GPU (TF32) jeśli dostępne
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def load_teacher_model(model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True, attn_impl=None):
    """Ładuje model nauczyciela; domyślnie 4-bit, opcjonalnie bf16, z wyborem impl. attention."""
    print(f"\n{'='*60}")
    print(f"ŁADOWANIE MODELU NAUCZYCIELA: {model_name}")
    print(f"{'='*60}")
    
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    print("Ładowanie tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    mode = "4-bit quantized" if load_in_4bit else "bf16 full precision"
    print(f"Ładowanie modelu ({mode})...")
    print("⚠️  To może zająć kilka minut...")
    
    model_kwargs = {
        "device_map": {"": 0},  # Wymuś cały model na GPU 0, uniknij CPU offload
        "torch_dtype": torch.float16 if load_in_4bit else torch.bfloat16,
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    model.eval()
    
    # Sprawdź VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"\n✓ Model załadowany")
        print(f"  VRAM używany: {allocated:.2f} GB")
        print(f"  VRAM zarezerwowany: {reserved:.2f} GB")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """Generuje odpowiedź dla pojedynczego prompta"""
    
    # Format jako chat
    messages = [{"role": "user", "content": prompt}]
    
    # Tokenizacja
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
    
    # Generowanie (greedy dla szybkości)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding - najszybsze
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # KV-cache znacząco przyspiesza generację
            num_beams=1  # Wyłącz beam search
        )
    
    # Dekodowanie
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # KLUCZOWE: Ekstremalnie agresywne czyszczenie
    del messages, input_text, inputs, outputs, generated_ids
    
    # Wymuś garbage collection natychmiast
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats - może pomóc z fragmentacją
        torch.cuda.reset_peak_memory_stats()
    
    return response, None

def process_dataset(
    model, 
    tokenizer, 
    dataset_path="../dataset.json",
    output_path="../teacher_responses.json",
    max_samples=None,
    temperature=0.7,
    max_new_tokens=512
):
    """Przetwarza cały dataset"""
    
    print(f"\n{'='*60}")
    print(f"GENEROWANIE ODPOWIEDZI NAUCZYCIELA")
    print(f"{'='*60}")
    
    # Wczytaj dataset
    print(f"\nŁadowanie datasetu: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Ograniczono do {max_samples} przykładów (test mode)")
    
    print(f"Znaleziono {len(dataset)} promptów")
    print(f"Temperatura: {temperature}")
    print(f"Max nowych tokenów: {max_new_tokens}")
    
    # Przetwarzanie
    results = []
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print("GENEROWANIE ODPOWIEDZI...")
    print(f"{'='*60}\n")
    
    for idx, item in enumerate(tqdm(dataset, desc="Generowanie", unit="prompt")):
        prompt = item['prompt']
        
        try:
            response, outputs = generate_response(
                model, 
                tokenizer, 
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            result = {
                "prompt": prompt,
                "response": response,
                "sheet": item.get('sheet', ''),
                "teacher_model": model.config._name_or_path,
                "temperature": temperature,
                "idx": idx
            }
            
            results.append(result)
            
            # Wyświetl przykład co 100 promptów
            if (idx + 1) % 100 == 0 or idx == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(dataset) - idx - 1) / rate if rate > 0 else 0
                
                print(f"\n{'─'*60}")
                print(f"Przykład {idx + 1}/{len(dataset)}:")
                print(f"{'─'*60}")
                print(f"Prompt (pierwsze 200 znaków):")
                print(f"  {prompt[:200]}...")
                print(f"\nOdpowiedź nauczyciela:")
                print(f"  {response[:300]}...")
                print(f"\nPostęp: {idx+1}/{len(dataset)}")
                print(f"Szybkość: {rate:.2f} prompt/s")
                print(f"Szacowany pozostały czas: {timedelta(seconds=int(remaining))}")
                
                # Monitoruj VRAM
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    print(f"VRAM używany: {allocated:.2f} GB")
                
                print(f"{'─'*60}\n")
            
            # EKSTREMALNIE agresywne czyszczenie co 5 promptów
            if (idx + 1) % 5 == 0:
                gc.collect()
                gc.collect()  # Podwójne dla pewności
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            
            # Monitoring fragmentacji co 20 promptów
            if (idx + 1) % 20 == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    print(f"\n[Memory Check #{idx+1}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Diff: {reserved-allocated:.2f}GB")
                
        except Exception as e:
            print(f"\n✗ Błąd przy promptzie {idx}: {e}")
            results.append({
                "prompt": prompt,
                "response": None,
                "error": str(e),
                "sheet": item.get('sheet', ''),
                "idx": idx
            })
    
    # Podsumowanie
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get('response') is not None)
    
    print(f"\n{'='*60}")
    print("PODSUMOWANIE GENEROWANIA")
    print(f"{'='*60}")
    print(f"Łącznie promptów: {len(dataset)}")
    print(f"Udanych odpowiedzi: {successful}")
    print(f"Błędów: {len(dataset) - successful}")
    print(f"Całkowity czas: {timedelta(seconds=int(total_time))}")
    print(f"Średni czas na prompt: {total_time/len(dataset):.2f}s")
    
    # Zapisz wyniki
    print(f"\nZapisywanie do: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Zapisano {len(results)} wyników")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generowanie odpowiedzi nauczyciela")
    parser.add_argument("--dataset", default="../dataset.json", help="Ścieżka do datasetu")
    parser.add_argument("--output", default="../teacher_responses.json", help="Ścieżka do outputu")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model nauczyciela")
    parser.add_argument("--no-4bit", action="store_true", help="Wyłącz 4-bit; użyj bf16 pełnego modelu (szybciej na 4090)")
    parser.add_argument("--attn-impl", default=None, help="Implementacja attention, np. flash_attention_2 jeśli zainstalowane")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit próbek (do testów)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura generowania")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max nowych tokenów")
    
    args = parser.parse_args()
    
    # Sprawdź CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA niedostępna! Ten skrypt wymaga GPU.")
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Załaduj model
    model, tokenizer = load_teacher_model(
        args.model,
        load_in_4bit=not args.no_4bit,
        attn_impl=args.attn_impl
    )
    
    # Przetwórz dataset
    results = process_dataset(
        model,
        tokenizer,
        dataset_path=args.dataset,
        output_path=args.output,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )
    
    print(f"\n{'='*60}")
    print("✓ FAZA 1 ZAKOŃCZONA")
    print(f"{'='*60}")
    print(f"Wygenerowano odpowiedzi nauczyciela: {args.output}")
    print("Możesz teraz uruchomić train_student.py")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

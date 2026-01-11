#!/usr/bin/env python3
"""
Phase 1: Teacher Response Generation (default: Qwen2.5 7B 4-bit)

Generates teacher model responses for all prompts in the dataset.
Supports both 4-bit quantized and full precision modes with aggressive memory
management to enable processing large datasets on consumer GPUs efficiently.
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
    """Loads the teacher model.
    
    Initializes the teacher model with configurable precision and attention implementation.
    Defaults to 4-bit quantization for memory efficiency, with optional full bf16 precision
    for faster inference on high-end GPUs. Forces full model placement on GPU 0 to avoid
    CPU offloading and maximize throughput.
    
    Args:
        model_name (str): Name or path of the teacher model.
            Defaults to "Qwen/Qwen2.5-7B-Instruct".
        load_in_4bit (bool): Whether to use 4-bit quantization. Defaults to True.
        attn_impl (str, optional): Attention implementation to use (e.g., "flash_attention_2").
            If None, uses default implementation.
    
    Returns:
        tuple: A tuple containing:
            - model: The loaded teacher model in evaluation mode.
            - tokenizer: The corresponding tokenizer.
    """
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
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"\n✓ Model załadowany")
        print(f"  VRAM używany: {allocated:.2f} GB")
        print(f"  VRAM zarezerwowany: {reserved:.2f} GB")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """Generates a response for a single prompt.
    
    Uses greedy decoding with KV-cache for fast inference and applies aggressive memory
    cleanup after each generation to prevent memory fragmentation during batch processing.
    All intermediate tensors are explicitly deleted and CUDA cache is cleared.
    
    Args:
        model: The teacher model to use for generation.
        tokenizer: The tokenizer for the model.
        prompt (str): The input prompt to generate a response for.
        max_new_tokens (int): Maximum number of tokens to generate. Defaults to 512.
        temperature (float): Sampling temperature (currently unused with greedy decoding).
            Defaults to 0.7.
    
    Returns:
        tuple: A tuple containing:
            - response (str): The generated response text.
            - None: Placeholder for compatibility (previously used for outputs).
    """
    
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
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    

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
    """Processes the entire dataset and generates teacher responses.
    
    Iterates through all prompts in the dataset, generates responses using the teacher model,
    and saves results to a JSON file. Implements periodic memory cleanup and progress monitoring
    with example outputs. Handles errors gracefully by recording failed prompts while continuing
    processing.
    
    Args:
        model: The teacher model to use for generation.
        tokenizer: The tokenizer for the model.
        dataset_path (str): Path to the input dataset JSON file.
            Defaults to "../dataset.json".
        output_path (str): Path to save the generated responses.
            Defaults to "../teacher_responses.json".
        max_samples (int, optional): Limit the number of samples to process (for testing).
            If None, processes all samples.
        temperature (float): Sampling temperature for generation. Defaults to 0.7.
        max_new_tokens (int): Maximum number of tokens to generate per response.
            Defaults to 512.
    
    Returns:
        list: List of result dictionaries containing prompts, responses, and metadata.
    """
    
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
    

    print(f"\nZapisywanie do: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Zapisano {len(results)} wyników")
    
    return results

def main():
    """Main entry point for teacher response generation.
    
    Parses command-line arguments, validates CUDA availability, loads the teacher model,
    and processes the dataset to generate responses for distillation training.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
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
    
    if not torch.cuda.is_available():
        print("✗ CUDA niedostępna! Ten skrypt wymaga GPU.")
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    model, tokenizer = load_teacher_model(
        args.model,
        load_in_4bit=not args.no_4bit,
        attn_impl=args.attn_impl
    )
    
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

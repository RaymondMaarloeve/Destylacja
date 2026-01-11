"""
Optimized Script for Teacher Model Response Generation

Leverages batching, bfloat16 precision, and scaled dot-product attention for 
efficient GPU inference. Designed for maximum throughput on modern GPU architectures
with aggressive memory management and TF32 optimizations enabled.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from typing import List, Dict
import gc
import argparse

# Konfiguracja domyślna
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "/root/destylacja/dataset.json"
OUTPUT_FILE = "/root/destylacja/qwen_responses.json"
BATCH_SIZE = 8  # Optymalne dla 48GB VRAM z modelem 7B
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# Włącz optymalizacje dla RTX 4090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_model_and_tokenizer():
    """Loads the model with GPU optimizations.
    
    Initializes the specified teacher model with bfloat16 precision and SDPA attention
    implementation. Configures left-padding for efficient batch generation with
    decoder-only models and monitors VRAM allocation.
    
    Returns:
        tuple: A tuple containing:
            - model: The loaded model in evaluation mode.
            - tokenizer: The configured tokenizer with padding setup.
    """
    print("🔄 Ładowanie modelu Qwen2.5-7B-Instruct...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side='left'  # Wymagane dla decoder-only models w batch generation
    )
    
    # Ustaw pad_token jeśli nie istnieje
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # Najlepsze dla Ada/Ampere
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # Scaled Dot Product Attention - szybkie i bez dodatkowych zależności
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    print(f"✅ Model załadowany na: {model.device}")
    print(f"💾 Wykorzystanie VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def load_dataset(file_path: str) -> List[Dict]:
    """Loads the dataset from a JSON file.
    
    Reads and parses the input JSON file containing prompts for batch processing.
    Validates file accessibility and reports the number of loaded prompts.
    
    Args:
        file_path (str): Path to the JSON dataset file.
    
    Returns:
        List[Dict]: List of dataset items, each containing at minimum a 'prompt' field.
    """
    print(f"📂 Wczytywanie datasetu z {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Wczytano {len(data)} promptów")
    return data


def prepare_messages(prompt: str) -> List[Dict]:
    """Prepares the prompt in chat format for the model.
    
    Converts a plain text prompt into the expected chat message format,
    creating a user role message suitable for the chat template.
    
    Args:
        prompt (str): The input prompt text.
    
    Returns:
        List[Dict]: A list containing a single message dictionary with role and content.
    """
    return [
        {"role": "user", "content": prompt}
    ]


def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    show_examples: bool = True
) -> List[str]:
    """Generates responses in batches for maximum efficiency.
    
    Processes multiple prompts simultaneously using batch inference with automatic
    padding and truncation. Implements KV-cache for faster generation and periodic
    memory cleanup to prevent VRAM fragmentation. Optionally displays sample outputs
    during processing for quality monitoring.
    
    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer for the model.
        prompts (List[str]): List of input prompts to process.
        batch_size (int): Number of prompts to process simultaneously.
            Defaults to BATCH_SIZE.
        max_new_tokens (int): Maximum tokens to generate per response.
            Defaults to MAX_NEW_TOKENS.
        show_examples (bool): Whether to display example outputs during processing.
            Defaults to True.
    
    Returns:
        List[str]: List of generated responses corresponding to input prompts.
    """
    
    all_responses = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print(f"🚀 Rozpoczynam generowanie w {num_batches} batch'ach...")
    
    with torch.no_grad():
        batch_num = 0
        for i in tqdm(range(0, len(prompts), batch_size), desc="Przetwarzanie"):
            batch_prompts = prompts[i:i + batch_size]
            
            # Przygotuj messages dla każdego promptu
            batch_messages = [prepare_messages(p) for p in batch_prompts]
            
            # Tokenizacja z padding dla batcha
            batch_texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for messages in batch_messages
            ]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            # Generowanie z optymalizacjami
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # KV cache dla szybszego generowania
            )
            
            # Dekodowanie odpowiedzi
            for j, output in enumerate(outputs):
                # Usuń input z outputu
                input_length = inputs['input_ids'][j].shape[0]
                response_ids = output[input_length:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                all_responses.append(response)
            
            # Wyświetl przykład co 5 batchów
            if show_examples and batch_num % 5 == 0 and len(all_responses) > 0:
                print("\n" + "="*80)
                print(f"📝 PRZYKŁAD (Batch {batch_num + 1}/{num_batches})")
                print("="*80)
                example_idx = len(all_responses) - 1
                print(f"\n🔵 INPUT (pierwsze 200 znaków):\n{batch_prompts[-1][:200]}...\n")
                print(f"🟢 OUTPUT:\n{all_responses[-1][:500]}\n")
                print("="*80 + "\n")
            
            batch_num += 1
            
            # Czyszczenie pamięci
            del inputs, outputs
            torch.cuda.empty_cache()
    
    return all_responses


def save_results(dataset: List[Dict], responses: List[str], output_file: str):
    """Saves grouped prompts with responses to JSON.
    
    Combines original dataset items with generated responses and writes them
    to a JSON file with proper UTF-8 encoding and readable formatting. Preserves
    metadata fields like sheet information from the original dataset.
    
    Args:
        dataset (List[Dict]): Original dataset items with prompts and metadata.
        responses (List[str]): Generated responses corresponding to each dataset item.
        output_file (str): Path where the results JSON file will be saved.
    """
    print(f"💾 Zapisywanie wyników do {output_file}...")
    
    results = []
    for item, response in zip(dataset, responses):
        results.append({
            "prompt": item["prompt"],
            "response": response,
            "sheet": item.get("sheet", "")
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Zapisano {len(results)} wyników")


def main():
    """Main entry point for teacher response generation.
    
    Orchestrates the complete pipeline: argument parsing, CUDA validation, dataset loading,
    model initialization, batch generation, and result saving. Provides comprehensive progress
    reporting and resource utilization monitoring throughout the process.
    """
    # Parsowanie argumentów
    parser = argparse.ArgumentParser(description='Generator odpowiedzi Qwen2.5-7B-Instruct')
    parser.add_argument('--input', default=INPUT_FILE, help='Plik wejściowy JSON')
    parser.add_argument('--output', default=OUTPUT_FILE, help='Plik wyjściowy JSON')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Rozmiar batcha')
    parser.add_argument('--max-tokens', type=int, default=MAX_NEW_TOKENS, help='Max tokenów do wygenerowania')
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 Generator odpowiedzi Qwen2.5-7B-Instruct dla RTX 4090")
    print("="*80)
    
    # Sprawdź CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA niedostępna! Wymagana GPU.")
    
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Całkowita VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Wczytaj dataset
    dataset = load_dataset(args.input)
    
    # Załaduj model
    model, tokenizer = load_model_and_tokenizer()
    print()
    
    # Ekstraktuj prompty
    prompts = [item["prompt"] for item in dataset]
    
    # Generuj odpowiedzi
    print(f"⚙️ Konfiguracja:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Temperature: {TEMPERATURE}")
    print(f"   Top-p: {TOP_P}")
    print(f"   Output: {args.output}")
    print()
    
    responses = generate_responses_batch(
        model, tokenizer, prompts, 
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        show_examples=True
    )
    
    # Zapisz wyniki
    save_results(dataset, responses, args.output)
    
    print()
    print("=" * 80)
    print(f"✨ Gotowe! Wyniki zapisane w: {args.output}")
    print(f"📊 Przetworzono: {len(responses)} promptów")
    print(f"💾 Finalne wykorzystanie VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("=" * 80)


if __name__ == "__main__":
    main()

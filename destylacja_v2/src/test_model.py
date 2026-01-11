#!/usr/bin/env python3
"""
Test wytrenowanego modelu studenta
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def load_model(model_path, base_model=None):
    """Ładuje wytrenowany model"""
    print(f"Ładowanie modelu z: {model_path}")
    
    # Jeśli to model LoRA, trzeba załadować base + adapter
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ Model załadowany (full model)")
    except:
        if base_model is None:
            base_model = "meta-llama/Llama-3.2-3B-Instruct"
        
        print(f"Ładowanie jako LoRA adapter (base: {base_model})")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base, model_path)
        print("✓ Model załadowany (LoRA)")
    
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_tokens=512, temperature=0.7):
    """Generuje odpowiedź"""
    messages = [{"role": "user", "content": prompt}]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response

def interactive_mode(model, tokenizer):
    """Tryb interaktywny"""
    print("\n" + "="*60)
    print("TRYB INTERAKTYWNY")
    print("="*60)
    print("Wpisz 'quit' aby zakończyć\n")
    
    while True:
        try:
            prompt = input("\n🎮 Prompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt.strip():
                continue
            
            print("\n🤖 Student odpowiada...")
            response = generate_response(model, tokenizer, prompt)
            print(f"\n{response}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nZakończono.")
            break
        except Exception as e:
            print(f"\nBłąd: {e}")

def test_examples(model, tokenizer):
    """Testuje przykładowe prompty"""
    examples = [
        "Hello how are you?",
        "What happened to you?",
        "What kind of city is this?",
        "Tell me about yourself",
        "What do you know about this place?"
    ]
    
    print("\n" + "="*60)
    print("TESTOWANIE PRZYKŁADOWYCH PROMPTÓW")
    print("="*60)
    
    for i, prompt in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Przykład {i}/{len(examples)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"\nOdpowiedź:")
        
        response = generate_response(model, tokenizer, prompt)
        print(response)
        print("\n")

def main():
    parser = argparse.ArgumentParser(description="Test wytrenowanego modelu")
    parser.add_argument("--model-path", default="../models/llama-3b-distilled",
                       help="Ścieżka do modelu")
    parser.add_argument("--base-model", default=None,
                       help="Base model (dla LoRA)")
    parser.add_argument("--interactive", action="store_true",
                       help="Tryb interaktywny")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Pojedynczy prompt do przetestowania")
    
    args = parser.parse_args()
    
    # Załaduj model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    if args.prompt:
        # Pojedynczy prompt
        print(f"\nPrompt: {args.prompt}")
        print("\nOdpowiedź:")
        response = generate_response(model, tokenizer, args.prompt)
        print(response)
    elif args.interactive:
        # Tryb interaktywny
        interactive_mode(model, tokenizer)
    else:
        # Testuj przykłady
        test_examples(model, tokenizer)

if __name__ == "__main__":
    main()

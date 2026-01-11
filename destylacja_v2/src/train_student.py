#!/usr/bin/env python3
"""
Phase 2: Student Training (Llama 3.2 3B) via Distillation

Fine-tuning with teacher responses using LoRA and supervised learning.
This script trains a smaller student model to mimic the behavior of a larger 
teacher model by learning from its generated responses, enabling efficient 
knowledge transfer while maintaining model quality.
"""
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm
import time
from datetime import timedelta
import argparse
import os

def load_student_model(model_name="meta-llama/Llama-3.2-3B-Instruct", use_lora=True):
    """Loads the student model with optional LoRA.
    
    Initializes the base student model with efficient memory usage via fp16 precision
    and applies LoRA adapters for parameter-efficient fine-tuning. Configures gradient
    checkpointing and monitors VRAM usage to optimize training on limited hardware.
    
    Args:
        model_name (str): Name or path of the base student model. 
            Defaults to "meta-llama/Llama-3.2-3B-Instruct".
        use_lora (bool): Whether to apply LoRA configuration. Defaults to True.
    
    Returns:
        tuple: A tuple containing:
            - model: The loaded (and optionally LoRA-adapted) model.
            - tokenizer: The corresponding tokenizer.
    """
    print(f"\n{'='*60}")
    print(f"ŁADOWANIE MODELU STUDENTA: {model_name}")
    print(f"{'='*60}")
    
    print("Ładowanie tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Ładowanie modelu...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    if use_lora:
        print("\nKonfigurowanie LoRA...")
        
        # Włącz gradient checkpointing jeśli nie jest włączony
        if not model.supports_gradient_checkpointing:
            print("⚠️  Model nie wspiera gradient checkpointing")
        else:
            model.gradient_checkpointing_enable()
        
        # Konfiguracja LoRA
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        # WAŻNE: Włącz input require grads dla gradient checkpointing
        model.enable_input_require_grads()
        
        model.print_trainable_parameters()
    
    # Sprawdź VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"\n✓ Model załadowany")
        print(f"  VRAM używany: {allocated:.2f} GB")
        print(f"  VRAM zarezerwowany: {reserved:.2f} GB")
    
    return model, tokenizer

def prepare_training_data(teacher_responses_path, tokenizer, max_length=1024):
    """Prepares training data from teacher responses.
    
    Loads teacher-generated responses, formats them into conversational chat templates,
    and tokenizes them for training. Automatically splits data into training and 
    evaluation sets while filtering out any invalid responses.
    
    Args:
        teacher_responses_path (str): Path to the JSON file containing teacher responses.
        tokenizer: The tokenizer to use for processing text.
        max_length (int): Maximum sequence length for tokenization. Defaults to 1024.
    
    Returns:
        tuple: A tuple containing:
            - train_dataset: The tokenized training dataset.
            - eval_dataset: The tokenized evaluation dataset.
            - formatted_data (list): List of formatted examples with prompts and responses.
    """
    print(f"\n{'='*60}")
    print("PRZYGOTOWANIE DANYCH TRENINGOWYCH")
    print(f"{'='*60}")
    
    # Wczytaj odpowiedzi nauczyciela
    print(f"\nŁadowanie: {teacher_responses_path}")
    with open(teacher_responses_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filtruj błędy
    valid_data = [d for d in data if d.get('response') is not None]
    print(f"Znaleziono {len(valid_data)} poprawnych przykładów (z {len(data)} total)")
    
    # Formatuj do konwersacji
    formatted_data = []
    for item in tqdm(valid_data, desc="Formatowanie danych"):
        prompt = item['prompt']
        response = item['response']
        
        # Format jako chat
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        # Aplikuj chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        formatted_data.append({
            "text": text,
            "prompt": prompt,
            "response": response
        })
    
    # Konwertuj do Dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenizacja
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    print("\nTokenizacja...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizacja"
    )
    
    # Split train/eval (90/10)
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"\n✓ Przygotowano dane:")
    print(f"  Trening: {len(train_dataset)} przykładów")
    print(f"  Ewaluacja: {len(eval_dataset)} przykładów")
    
    return train_dataset, eval_dataset, formatted_data

class ProgressCallback:
    """Callback for displaying training progress with sample responses.
    
    Monitors training metrics and periodically generates sample responses to evaluate
    the model's evolving capabilities during the distillation process. Provides real-time
    feedback on training speed, estimated completion time, and model quality.
    
    Attributes:
        model: The model being trained.
        tokenizer: The tokenizer for the model.
        test_prompts (list): List of test prompts for generating sample responses.
        total_steps (int): Total number of training steps.
    """
    def __init__(self, model, tokenizer, test_prompts, total_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step = state.global_step
        
        # Wyświetl progres co 50 kroków
        if self.step % 50 == 0:
            elapsed = time.time() - self.start_time
            rate = self.step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - self.step
            remaining_time = remaining_steps / rate if rate > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"POSTĘP TRENINGU - Krok {self.step}/{self.total_steps}")
            print(f"{'='*60}")
            print(f"Czas treningu: {timedelta(seconds=int(elapsed))}")
            print(f"Szacowany pozostały czas: {timedelta(seconds=int(remaining_time))}")
            print(f"Szybkość: {rate:.2f} kroków/s")
            
            # Wygeneruj przykładowe odpowiedzi
            if self.step % 200 == 0 and self.step > 0:
                self.generate_samples()
    
    def generate_samples(self):
        """Generates sample responses from the model during training."""
        print(f"\n{'─'*60}")
        print("PRZYKŁADOWE ODPOWIEDZI STUDENTA:")
        print(f"{'─'*60}")
        
        self.model.eval()
        
        for i, prompt in enumerate(self.test_prompts[:2]):  # Tylko 2 przykłady
            print(f"\nPrzykład {i+1}:")
            print(f"Prompt: {prompt[:150]}...")
            
            try:
                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                print(f"Odpowiedź: {response[:250]}...")
                
            except Exception as e:
                print(f"Błąd generowania: {e}")
        
        print(f"{'─'*60}\n")
        self.model.train()

def train_student(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    test_prompts,
    output_dir="../models/llama-3b-distilled",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    gradient_accumulation_steps=4
):
    """Trains the student model using knowledge distillation.
    
    Executes the full training pipeline with supervised learning on teacher responses.
    Implements gradient accumulation for effective large batch training, cosine learning
    rate scheduling, and periodic evaluation with checkpointing to ensure optimal results.
    
    Args:
        model: The student model to train.
        tokenizer: The tokenizer for the model.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        test_prompts (list): List of prompts for generating sample responses during training.
        output_dir (str): Directory to save the trained model. 
            Defaults to "../models/llama-3b-distilled".
        num_epochs (int): Number of training epochs. Defaults to 3.
        batch_size (int): Batch size per device. Defaults to 4.
        learning_rate (float): Learning rate for training. Defaults to 5e-5.
        gradient_accumulation_steps (int): Number of gradient accumulation steps. 
            Defaults to 4.
    
    Returns:
        Trainer: The trained Trainer object.
    """
    print(f"\n{'='*60}")
    print("KONFIGURACJA TRENINGU")
    print(f"{'='*60}")
    
    total_steps = (len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs
    
    print(f"Epoki: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Szacowana liczba kroków: {total_steps}")
    print(f"Output dir: {output_dir}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=200,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,  # Wyłączone bo LoRA ma własne
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        save_total_limit=3,
        report_to="none",
        disable_tqdm=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Callback
    progress_callback = ProgressCallback(model, tokenizer, test_prompts, total_steps)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Custom callback integration
    class CustomCallback(torch.nn.Module):
        def on_step_end(self, args, state, control, **kwargs):
            progress_callback.on_step_end(args, state, control, **kwargs)
    
    # Trening
    print(f"\n{'='*60}")
    print("ROZPOCZĘCIE TRENINGU")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        trainer.train()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("✓ TRENING ZAKOŃCZONY")
        print(f"{'='*60}")
        print(f"Całkowity czas: {timedelta(seconds=int(total_time))}")
        print(f"Finalny eval loss: {trainer.state.log_history[-1].get('eval_loss', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Trening przerwany przez użytkownika")
        print("Zapisywanie checkpointu...")
    
    # Zapisz finalny model
    print(f"\nZapisywanie modelu do: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Model zapisany")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Trening studenta przez destylację")
    parser.add_argument("--teacher-responses", default="../teacher_responses.json", 
                       help="Ścieżka do odpowiedzi nauczyciela")
    parser.add_argument("--student-model", default="meta-llama/Llama-3.2-3B-Instruct",
                       help="Model bazowy studenta")
    parser.add_argument("--output-dir", default="../models/llama-3b-distilled",
                       help="Katalog output")
    parser.add_argument("--epochs", type=int, default=3, help="Liczba epok")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=2, 
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--no-lora", action="store_true", help="Wyłącz LoRA")
    parser.add_argument("--max-length", type=int, default=1024, help="Max długość sekwencji")
    
    args = parser.parse_args()
    
    # Sprawdź CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA niedostępna! Ten skrypt wymaga GPU.")
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Załaduj model studenta
    model, tokenizer = load_student_model(args.student_model, use_lora=not args.no_lora)
    
    # Przygotuj dane
    train_dataset, eval_dataset, formatted_data = prepare_training_data(
        args.teacher_responses,
        tokenizer,
        max_length=args.max_length
    )
    
    test_prompts = [d['prompt'] for d in formatted_data[:5]]
    
    trainer = train_student(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        test_prompts,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation
    )
    
    print(f"\n{'='*60}")
    print("✓ FAZA 2 ZAKOŃCZONA")
    print(f"{'='*60}")
    print(f"Model studenta zapisany w: {args.output_dir}")
    print("\nMożesz teraz użyć wytrenowanego modelu!")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

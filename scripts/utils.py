"""
Utility functions for LLM to NPC distillation.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    path: str,
    metadata: Optional[Dict] = None
):
    """Zapisuje checkpoint modelu."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, path)
    print(f"Checkpoint zapisany: {path}")


def load_checkpoint(model, optimizer, path: str):
    """Ładuje checkpoint modelu."""
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def count_parameters(model) -> Dict[str, int]:
    """Zlicza parametry modelu."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6,
    }


def format_prompt_llama2(system: str, user: str) -> str:
    """Formatuje prompt dla Llama-2."""
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def format_prompt_mistral(system: str, user: str) -> str:
    """Formatuje prompt dla Mistral."""
    return f"<s>[INST] {system}\n\n{user} [/INST]"


def format_prompt_generic(system: str, user: str) -> str:
    """Formatuje prompt - format generyczny."""
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def get_prompt_formatter(model_name: str):
    """Zwraca odpowiednią funkcję formatującą dla danego modelu."""
    model_name_lower = model_name.lower()
    
    if "llama-2" in model_name_lower or "llama2" in model_name_lower:
        return format_prompt_llama2
    elif "mistral" in model_name_lower:
        return format_prompt_mistral
    else:
        return format_prompt_generic


def calculate_model_size_mb(model) -> float:
    """Oblicza rozmiar modelu w MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def load_jsonl(file_path: str) -> List[Dict]:
    """Ładuje dane z pliku JSONL."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Zapisuje dane do pliku JSONL."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def truncate_text(text: str, max_length: int = 100) -> str:
    """Skraca tekst do maksymalnej długości."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def print_model_info(model, model_name: str = "Model"):
    """Wyświetla informacje o modelu."""
    print(f"\n{'='*60}")
    print(f"{model_name} Info")
    print(f"{'='*60}")
    
    params = count_parameters(model)
    print(f"Całkowite parametry: {params['total_millions']:.2f}M")
    print(f"Trenowalne parametry: {params['trainable_millions']:.2f}M")
    print(f"Zamrożone parametry: {params['frozen'] / 1e6:.2f}M")
    
    size_mb = calculate_model_size_mb(model)
    print(f"Rozmiar modelu: {size_mb:.2f} MB")
    print(f"{'='*60}\n")


class TokenCounter:
    """Licznik tokenów dla monitorowania."""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def add_prompt(self, num_tokens: int):
        """Dodaje tokeny z promptu."""
        self.prompt_tokens += num_tokens
        self.total_tokens += num_tokens
    
    def add_completion(self, num_tokens: int):
        """Dodaje tokeny z odpowiedzi."""
        self.completion_tokens += num_tokens
        self.total_tokens += num_tokens
    
    def get_stats(self) -> Dict[str, int]:
        """Zwraca statystyki."""
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
        }
    
    def reset(self):
        """Resetuje licznik."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

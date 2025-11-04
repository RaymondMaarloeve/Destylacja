"""
Skrypt do testowania i ewaluacji wydestylowanego modelu.
Porównuje jakość odpowiedzi studenta z nauczycielem.
"""

import torch
import yaml
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


class ModelEvaluator:
    """Ewaluator porównujący wydestylowany model z nauczycielem."""
    
    def __init__(
        self,
        student_model_path: str,
        teacher_model_name: str,
        config_path: str = "config.yaml"
    ):
        """Inicjalizacja ewaluatora."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Ładowanie modeli do ewaluacji...")
        
        # Ładowanie tokenizera
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ładowanie modelu studenta
        print(f"  Ładowanie studenta z {student_model_path}")
        
        # Sprawdź który model bazowy użyć
        student_base_name = self.config['student_model']['name']
        
        base_model = AutoModelForCausalLM.from_pretrained(
            student_base_name,  # Używamy z config!
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.student_model = PeftModel.from_pretrained(base_model, student_model_path)
        self.student_model.eval()
        
        # Ładowanie modelu nauczyciela (opcjonalnie)
        if teacher_model_name:
            print(f"  Ładowanie nauczyciela {teacher_model_name}")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )
            self.teacher_model.eval()
        else:
            self.teacher_model = None
        
        print("✓ Modele załadowane")
    
    @torch.no_grad()
    def generate_response(
        self,
        model,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generuje odpowiedź z danego modelu."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def calculate_perplexity(self, model, text: str) -> float:
        """Oblicza perplexity dla danego tekstu."""
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = 512
        
        nlls = []
        for i in range(0, encodings.input_ids.size(1), max_length):
            begin_loc = max(i + max_length - 1024, 0)
            end_loc = min(i + max_length, encodings.input_ids.size(1))
            trg_len = end_loc - i
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.item()
    
    def compare_models(self, test_prompts: List[str], output_path: str = "evaluation_results.json"):
        """Porównuje odpowiedzi studenta i nauczyciela."""
        results = []
        
        print(f"\nPorównywanie modeli na {len(test_prompts)} promptach...\n")
        
        for prompt in tqdm(test_prompts):
            # Generuj z obu modeli
            student_response = self.generate_response(self.student_model, prompt)
            
            teacher_response = None
            if self.teacher_model:
                teacher_response = self.generate_response(self.teacher_model, prompt)
            
            # Oblicz metryki
            student_ppl = self.calculate_perplexity(self.student_model, prompt + " " + student_response)
            
            result = {
                'prompt': prompt,
                'student_response': student_response,
                'teacher_response': teacher_response,
                'student_perplexity': student_ppl,
                'student_length': len(student_response.split()),
            }
            
            if teacher_response:
                result['teacher_length'] = len(teacher_response.split())
            
            results.append(result)
        
        # Zapisz wyniki
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Podsumowanie
        print("\n" + "=" * 60)
        print("WYNIKI EWALUACJI")
        print("=" * 60)
        
        avg_student_ppl = np.mean([r['student_perplexity'] for r in results])
        avg_student_len = np.mean([r['student_length'] for r in results])
        
        print(f"\nModel Student:")
        print(f"  Średnie perplexity: {avg_student_ppl:.2f}")
        print(f"  Średnia długość odpowiedzi: {avg_student_len:.1f} słów")
        
        if self.teacher_model:
            avg_teacher_len = np.mean([r['teacher_length'] for r in results])
            print(f"\nModel Teacher:")
            print(f"  Średnia długość odpowiedzi: {avg_teacher_len:.1f} słów")
        
        print(f"\n✓ Pełne wyniki zapisane w: {output_path}")
        
        return results
    
    def interactive_test(self):
        """Interaktywny tryb testowania modelu."""
        print("\n" + "=" * 60)
        print("INTERAKTYWNY TEST MODELU")
        print("=" * 60)
        print("Wpisz 'quit' aby wyjść\n")
        
        while True:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print("\n🤖 Student:")
            student_response = self.generate_response(self.student_model, prompt)
            print(f"  {student_response}")
            
            if self.teacher_model:
                print("\n👨‍🏫 Teacher:")
                teacher_response = self.generate_response(self.teacher_model, prompt)
                print(f"  {teacher_response}")
            
            print()


def create_test_prompts() -> List[str]:
    """Tworzy zestaw testowych promptów dla różnych scenariuszy NPC."""
    # Załaduj prompty testowe z JSON
    npc_data_path = Path(__file__).parent.parent / "npc_data.json"
    
    if not npc_data_path.exists():
        print(f"⚠️  Plik {npc_data_path} nie istnieje, używam domyślnych promptów")
        return _create_default_test_prompts()
    
    with open(npc_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    for test_case in data['test_prompts']:
        # Format dla Llama-2 (dostosuj jeśli używasz innego modelu)
        prompt = f"<s>[INST] <<SYS>>\n{test_case['context']}\n<</SYS>>\n\n"
        prompt += f"Sytuacja: {test_case['situation']}\n"
        prompt += f"Gracz: {test_case['player_message']} [/INST]"
        prompts.append(prompt)
    
    print(f"✓ Załadowano {len(prompts)} testowych promptów z {npc_data_path}")
    return prompts


def _create_default_test_prompts() -> List[str]:
    """Tworzy domyślne prompty testowe jako fallback."""
    return [
        # Kupiec
        "<s>[INST] <<SYS>>\nJesteś kupcem w małym miasteczku. Sprzedajesz różne towary i uwielbiasz targować się.\n<</SYS>>\n\nSytuacja: Gracz pyta o ceny\nGracz: Ile kosztuje ten miecz? [/INST]",
        
        # Strażnik
        "<s>[INST] <<SYS>>\nJesteś strażnikiem miejskim. Dbasz o porządek i bezpieczeństwo.\n<</SYS>>\n\nSytuacja: Gracz pyta o drogę\nGracz: Jak dostać się do zamku? [/INST]",
    ]


def main():
    """Główna funkcja ewaluacji."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ewaluacja wydestylowanego modelu')
    parser.add_argument(
        '--student_path',
        type=str,
        default='./models/distilled_npc',
        help='Ścieżka do wydestylowanego modelu'
    )
    parser.add_argument(
        '--teacher_name',
        type=str,
        default=None,
        help='Nazwa modelu nauczyciela (opcjonalne)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['compare', 'interactive'],
        default='interactive',
        help='Tryb ewaluacji'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Ścieżka do pliku konfiguracyjnego'
    )
    
    args = parser.parse_args()
    
    # Inicjalizacja ewaluatora
    evaluator = ModelEvaluator(
        student_model_path=args.student_path,
        teacher_model_name=args.teacher_name,
        config_path=args.config
    )
    
    if args.mode == 'compare':
        # Porównanie na testowych promptach
        test_prompts = create_test_prompts()
        evaluator.compare_models(test_prompts)
    else:
        # Interaktywny test
        evaluator.interactive_test()


if __name__ == "__main__":
    main()

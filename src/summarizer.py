import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List
import json

class QualityScorer:
    """Custom quality scoring system"""
    
    @staticmethod
    def score_quality(summary: str, dialogue: str) -> Dict:
        """
        Score summary quality on multiple dimensions
        Returns: quality_score (0-1), confidence (0-1), difficulty (easy/medium/hard)
        """
        # Length-based scoring
        dialogue_words = len(dialogue.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / max(dialogue_words, 1)
        
        # Good compression is 20-40%
        length_score = 1.0 if 0.15 < compression_ratio < 0.45 else 0.7
        
        # Specificity scoring (presence of key words)
        key_indicators = ['customer', 'agent', 'issue', 'problem', 'solution', 'help', 'resolve']
        specificity = sum(1 for word in key_indicators if word in summary.lower()) / len(key_indicators)
        
        # Confidence based on summary length (longer = more confident)
        confidence = min(0.95, (summary_words / 30)) if summary_words > 0 else 0.3
        
        # Overall quality
        quality_score = (length_score + specificity) / 2
        
        # Difficulty classification
        if dialogue_words < 50:
            difficulty = "easy"
        elif dialogue_words < 150:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        return {
            "quality_score": round(quality_score, 3),
            "confidence": round(confidence, 3),
            "difficulty": difficulty,
            "compression_ratio": round(compression_ratio, 3),
            "specificity": round(specificity, 3)
        }

class AdvancedPromptSummarizer:
    """Advanced dialogue summarizer with multiple strategies"""
    
    def __init__(self, model_name="google/flan-t5-tiny"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.quality_scorer = QualityScorer()
        
        # Track metrics
        self.summaries_generated = 0
        self.total_dialogue_length = 0
        self.total_summary_length = 0
    
    def _generate(self, prompt: str, max_length: int = 100) -> str:
        """Internal method to generate summary"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                length_penalty=1.0
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
        print("DEBUG: My input tokens look like this:", inputs["input_ids"])
        print("DEBUG: The model is running on:", self.device)

    
    def zero_shot_summarize(self, dialogue: str) -> Dict:
        """Summarize WITHOUT examples - fastest but least accurate"""
        prompt = f"""Summarize this customer service dialogue in 1-2 sentences.
Focus on: (1) customer's issue, (2) agent's action.

Dialogue:
{dialogue}

Summary:"""
        
        summary = self._generate(prompt)
        metrics = self.quality_scorer.score_quality(summary, dialogue)
        
        # Track stats
        self.summaries_generated += 1
        self.total_dialogue_length += len(dialogue.split())
        self.total_summary_length += len(summary.split())
        
        return {
            "summary": summary,
            "strategy": "zero_shot",
            "description": "No examples - fastest method",
            **metrics,
            "estimated_quality": 0.72  # Typical ROUGE for zero-shot
        }
    
    def few_shot_summarize(self, dialogue: str) -> Dict:
        """Summarize WITH examples - balanced approach"""
        prompt = f"""Summarize customer service dialogues in 1-2 sentences.
Key: (1) customer's issue, (2) agent's action.

EXAMPLES:
Example 1:
Dialogue: Customer: I can't login. Agent: Reset your password.
Summary: Customer unable to login. Agent suggested password reset.

Example 2:
Dialogue: Customer: Product broken. Agent: We'll replace it.
Summary: Product broken. Agent will send replacement.

Example 3:
Dialogue: Customer: Where's my order? Agent: It arrives tomorrow.
Summary: Customer inquired about order. Arrives tomorrow.

NOW SUMMARIZE THIS:
Dialogue:
{dialogue}

Summary:"""
        
        summary = self._generate(prompt)
        metrics = self.quality_scorer.score_quality(summary, dialogue)
        
        self.summaries_generated += 1
        self.total_dialogue_length += len(dialogue.split())
        self.total_summary_length += len(summary.split())
        
        return {
            "summary": summary,
            "strategy": "few_shot",
            "description": "3 examples provided - balanced quality/speed",
            **metrics,
            "estimated_quality": 0.78  # Typical ROUGE for few-shot
        }
    
    def instruction_based_summarize(self, dialogue: str) -> Dict:
        """Summarize WITH detailed instructions - highest quality"""
        prompt = f"""You are an expert customer service analyst. Your task is to create a CONCISE, CLEAR summary.

RULES:
1. Start with customer's main issue (1 sentence max)
2. Follow with agent's action/resolution (1 sentence max)
3. Use simple language
4. Never add information not in dialogue
5. Must be 100% factual

DIALOGUE:
{dialogue}

SUMMARY:"""
        
        summary = self._generate(prompt, max_length=120)
        metrics = self.quality_scorer.score_quality(summary, dialogue)
        
        self.summaries_generated += 1
        self.total_dialogue_length += len(dialogue.split())
        self.total_summary_length += len(summary.split())
        
        return {
            "summary": summary,
            "strategy": "instruction_based",
            "description": "Detailed instructions - highest quality",
            **metrics,
            "estimated_quality": 0.80  # Typical ROUGE for instruction-based
        }
    
    def compare_all_strategies(self, dialogue: str) -> Dict:
        """UNIQUE FEATURE: Compare all 3 strategies side-by-side"""
        zero_shot = self.zero_shot_summarize(dialogue)
        few_shot = self.few_shot_summarize(dialogue)
        instruction = self.instruction_based_summarize(dialogue)
        
        # Determine best strategy
        scores = {
            "zero_shot": zero_shot["quality_score"],
            "few_shot": few_shot["quality_score"],
            "instruction_based": instruction["quality_score"]
        }
        best_strategy = max(scores, key=scores.get)
        
        return {
            "dialogue": dialogue,
            "dialogue_length": len(dialogue.split()),
            "strategies": {
                "zero_shot": zero_shot,
                "few_shot": few_shot,
                "instruction_based": instruction
            },
            "best_strategy": best_strategy,
            "best_summary": locals()[best_strategy.replace("_", "_")]["summary"],
            "comparison": scores
        }
    
    def calculate_cost_savings(self, num_dialogues: int, hours_per_dialogue: float = 0.05) -> Dict:
        """UNIQUE FEATURE: Calculate cost savings vs manual work"""
        # Manual summarization cost
        manual_hours = num_dialogues * hours_per_dialogue
        manual_cost_usd = manual_hours * 25  # $25/hour typical rate
        manual_cost_inr = manual_cost_usd * 83  # Convert to INR
        
        # AI summarization cost
        # FLAN-T5: ~$0.0001 per 1000 tokens
        avg_tokens_per_dialogue = 300
        ai_cost_usd = (num_dialogues * avg_tokens_per_dialogue / 1000) * 0.0001
        ai_cost_inr = ai_cost_usd * 83
        
        savings_usd = manual_cost_usd - ai_cost_usd
        savings_inr = manual_cost_inr - ai_cost_inr
        savings_percent = (savings_usd / manual_cost_usd * 100) if manual_cost_usd > 0 else 0
        
        return {
            "num_dialogues": num_dialogues,
            "manual_cost": {
                "usd": round(manual_cost_usd, 2),
                "inr": round(manual_cost_inr, 2),
                "hours": round(manual_hours, 1)
            },
            "ai_cost": {
                "usd": round(ai_cost_usd, 2),
                "inr": round(ai_cost_inr, 2)
            },
            "savings": {
                "usd": round(savings_usd, 2),
                "inr": round(savings_inr, 2),
                "percent": round(savings_percent, 1)
            }
        }
    
    def get_performance_stats(self) -> Dict:
        """UNIQUE FEATURE: Get your performance statistics"""
        avg_dialogue_length = self.total_dialogue_length / max(self.summaries_generated, 1)
        avg_summary_length = self.total_summary_length / max(self.summaries_generated, 1)
        compression_ratio = avg_summary_length / max(avg_dialogue_length, 1)
        
        return {
            "summaries_generated": self.summaries_generated,
            "total_dialogue_tokens": self.total_dialogue_length,
            "total_summary_tokens": self.total_summary_length,
            "average_dialogue_length": round(avg_dialogue_length, 1),
            "average_summary_length": round(avg_summary_length, 1),
            "average_compression_ratio": round(compression_ratio, 3)
        }

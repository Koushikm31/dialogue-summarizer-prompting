from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(
    title="Dialogue Summarizer API",
    description="Summarize conversations using local optimized FLAN-T5-tiny",
    version="2.0"
)

print("Loading model on startup...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-tiny")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-tiny")
device = torch.device("cpu")
model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")

class SummarizeRequest(BaseModel):
    dialogue: str
    strategy: str = "few_shot"

@app.post("/summarize")
def summarize_dialogue(request: SummarizeRequest):
    """Summarize using local optimized model"""
    try:
        if not request.dialogue or len(request.dialogue) < 10:
            return {"error": "Dialogue too short", "status": 400}
        
        # Simple, fast prompt
        prompt = f"Summarize this dialogue:\n{request.dialogue}\nSummary:"
        
        # Tokenize with memory optimization
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(device)
        
        # Generate with optimizations
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=60,
                num_beams=1,  # Beam search disabled for speed
                temperature=0.7
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "summary": summary.strip(),
            "strategy": request.strategy,
            "status": 200
        }
    
    except Exception as e:
        return {"error": str(e), "status": 500}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "google/flan-t5-tiny (237MB, optimized)",
        "version": "2.0"
    }

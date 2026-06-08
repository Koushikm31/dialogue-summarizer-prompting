from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(
    title="Dialogue Summarizer API",
    description="Summarize customer-agent conversations using HuggingFace Inference API",
    version="2.0"
)

# Get HF token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

class SummarizeRequest(BaseModel):
    dialogue: str
    strategy: str = "few_shot"

def call_hf_api(prompt: str) -> str:
    """Call HuggingFace Inference API"""
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        return f"Error: {response.status_code}"
    
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "No summary generated")
    return "No summary generated"

@app.post("/summarize")
def summarize_dialogue(request: SummarizeRequest):
    """Main endpoint - summarize dialogue"""
    try:
        if not request.dialogue or len(request.dialogue) < 10:
            return {"error": "Dialogue too short (min 10 characters)", "status": 400}
        
        # Create prompt based on strategy
        if request.strategy == "few_shot":
            prompt = f"""Summarize this dialogue in 1-2 sentences.
Example: Customer: I want to cancel. Agent: Why?
Summary: Customer wants to cancel.

Dialogue: {request.dialogue}
Summary:"""
        else:
            prompt = f"""Summarize: {request.dialogue}"""
        
        summary = call_hf_api(prompt)
        
        return {
            "summary": summary,
            "strategy": request.strategy,
            "status": 200
        }
    except Exception as e:
        return {"error": str(e), "status": 500}

@app.get("/health")
def health_check():
    """Check if server is running"""
    return {
        "status": "healthy",
        "model": "google/flan-t5-small (via HF API)",
        "version": "2.0"
    }

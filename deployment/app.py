from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.summarizer import AdvancedPromptSummarizer

# Initialize FastAPI
app = FastAPI(
    title="Advanced Dialogue Summarizer API",
    description="Summarize dialogues with quality scoring, confidence levels, and strategy comparison",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
print("[INFO] Loading FLAN-T5 model...")
summarizer = AdvancedPromptSummarizer(model_name="google/flan-t5-base")
print("[INFO] Model loaded successfully!")

# Request/Response models
class SummarizeRequest(BaseModel):
    dialogue: str
    strategy: str = "few_shot"  # zero_shot, few_shot, instruction_based, compare_all

class SummarizeResponse(BaseModel):
    summary: str
    strategy: str
    quality_score: float
    confidence: float
    difficulty: str

# API Routes
@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Advanced Dialogue Summarizer API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model": "google/flan-t5-base",
        "api_version": "2.0.0",
        "features": [
            "zero_shot_summarization",
            "few_shot_summarization",
            "instruction_based_summarization",
            "quality_scoring",
            "confidence_estimation",
            "strategy_comparison",
            "cost_savings_calculation"
        ]
    }

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Summarize a dialogue using specified strategy
    
    Strategies:
    - zero_shot: No examples (fastest, ~72% quality)
    - few_shot: 3 examples (balanced, ~78% quality)
    - instruction_based: Detailed instructions (best, ~80% quality)
    - compare_all: Try all strategies and return the best one
    """
    try:
        if not request.dialogue or len(request.dialogue.strip()) == 0:
            raise HTTPException(status_code=400, detail="Dialogue cannot be empty")
        
        if request.strategy == "zero_shot":
            result = summarizer.zero_shot_summarize(request.dialogue)
        elif request.strategy == "few_shot":
            result = summarizer.few_shot_summarize(request.dialogue)
        elif request.strategy == "instruction_based":
            result = summarizer.instruction_based_summarize(request.dialogue)
        elif request.strategy == "compare_all":
            result = summarizer.compare_all_strategies(request.dialogue)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {request.strategy}. Use: zero_shot, few_shot, instruction_based, or compare_all"
            )
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost-calculator")
async def cost_calculator(num_dialogues: int = 10000):
    """
    UNIQUE ENDPOINT: Calculate cost savings for your use case
    
    Parameters:
    - num_dialogues: How many dialogues you need to process
    
    Returns: Manual vs AI costs and total savings
    """
    try:
        if num_dialogues < 1:
            raise HTTPException(status_code=400, detail="num_dialogues must be >= 1")
        
        result = summarizer.calculate_cost_savings(num_dialogues)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """UNIQUE ENDPOINT: Get API performance statistics"""
    try:
        stats = summarizer.get_performance_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-summarize")
async def batch_summarize(dialogues: List[str], strategy: str = "few_shot"):
    """
    UNIQUE ENDPOINT: Summarize multiple dialogues at once
    
    Perfect for batch processing
    """
    try:
        from typing import List
        
        if not dialogues:
            raise HTTPException(status_code=400, detail="Dialogues list cannot be empty")
        
        results = []
        for dialogue in dialogues:
            if strategy == "zero_shot":
                result = summarizer.zero_shot_summarize(dialogue)
            elif strategy == "few_shot":
                result = summarizer.few_shot_summarize(dialogue)
            elif strategy == "instruction_based":
                result = summarizer.instruction_based_summarize(dialogue)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
            results.append(result)
        
        return JSONResponse(content={
            "total_dialogues": len(dialogues),
            "strategy_used": strategy,
            "summaries": results
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

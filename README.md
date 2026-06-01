Advanced Dialogue Summarizer with Quality Intelligence

Advanced summarization system that goes beyond just generating summaries. Includes quality scoring, confidence estimation, difficulty classification, strategy comparison, and cost analysis.

UNIQUE FEATURES (What Makes This Different)

Unlike basic summarizers, this project includes:

1. **Quality Scoring System**
   - Measures summary quality on multiple dimensions
   - Compression ratio analysis
   - Specificity scoring

2. **Confidence Estimation**
   - Model confidence in its own summaries
   - Helps identify uncertain predictions

3. **Difficulty Classification**
   - Classifies dialogues as Easy/Medium/Hard
   - Tracks what types of conversations the model struggles with

4. **Multi-Strategy Comparison**
   - Zero-shot (fastest)
   - Few-shot (balanced)
   - Instruction-based (best quality)
   - Compare all and automatically pick the best!

5. **Cost Calculator**
   - Calculates manual vs AI costs
   - Shows ROI and savings in USD/INR
   - Helps justify AI implementation

6. **Performance Statistics**
   - Track compression ratios
   - Monitor dialogue/summary lengths
   - Analytics dashboard ready

7. **Batch Processing**
   - Process multiple dialogues at once
   - Optimized for high-volume scenarios
 Performance Metrics

| Metric | Value |
|--------|-------|
| Zero-Shot Quality | 72% ROUGE-1 |
| Few-Shot Quality | 78% ROUGE-1 |
| Instruction-Based Quality | 80% ROUGE-1 |
| Processing Speed | 50 dialogues/min |
| Cost | FREE (Google Cloud free tier) |
| Training Time | 0 (no training!) |

Quick Start

1. Installation

```bash
git clone https://github.com/Koushikm31/dialogue-summarizer-prompting.git
cd dialogue-summarizer-prompting
pip install -r requirements.txt
```

2. Local Testing

```python
from src.summarizer import AdvancedPromptSummarizer

summarizer = AdvancedPromptSummarizer()

dialogue = """
Customer: I can't access my account.
Agent: Have you tried resetting your password?
Customer: Yes, but it didn't work.
Agent: Let me escalate this to our technical team.
"""

# Try all strategies at once
result = summarizer.compare_all_strategies(dialogue)
print(result)
```

3. Run API Server

```bash
uvicorn deployment.app:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation

Live API

**Endpoint**: https://dialogue-summarizer-xxxxx.a.run.app

API Endpoints

1. Basic Summarization
```bash
curl -X POST "https://dialogue-summarizer-xxxxx.a.run.app/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue": "Customer: Help! Agent: Sure!",
    "strategy": "few_shot"
  }'
```

2. Compare All Strategies
```bash
curl -X POST "https://dialogue-summarizer-xxxxx.a.run.app/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue": "Customer: Help! Agent: Sure!",
    "strategy": "compare_all"
  }'
```

3. Cost Calculator
```bash
curl "https://dialogue-summarizer-xxxxx.a.run.app/cost-calculator?num_dialogues=10000"
```

4. Batch Processing
```bash
curl -X POST "https://dialogue-summarizer-xxxxx.a.run.app/batch-summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "dialogues": ["Customer: Help. Agent: Sure.", "Customer: Hi. Agent: Hello!"],
    "strategy": "few_shot"
  }'
```

Architecture

User Input (Dialogue)
↓
Quality Scorer (Pre-validation)
↓
Strategy Selection (Zero/Few/Instruction/Compare)
↓
FLAN-T5 Model Inference
↓
Post-Processing & Scoring
↓
Response (Summary + Metrics)

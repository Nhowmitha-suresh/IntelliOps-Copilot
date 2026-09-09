import os
import csv
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from generate import load_model_and_tokenizer, generate_text, generate_instruction


def resolve_path(rel_path: str) -> str:
    """
    Resolves relative path whether executing from root or intelliops-copilot directory.
    """
    if os.path.exists(rel_path):
        return rel_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_from_base = os.path.join(base_dir, rel_path)
    if os.path.exists(path_from_base):
        return path_from_base

    path_with_prefix = os.path.join("intelliops-copilot", rel_path)
    if os.path.exists(path_with_prefix):
        return path_with_prefix

    return path_from_base


MODELS: Dict[str, Any] = {}
TOKENIZER = None


def get_last_val_perplexity(csv_rel_path: str) -> float:
    full_path = resolve_path(csv_rel_path)
    if not os.path.exists(full_path):
        return 0.0
    val_ppl = 0.0
    try:
        with open(full_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "val_ppl" in row and row["val_ppl"]:
                    val_ppl = float(row["val_ppl"])
    except Exception:
        pass
    return val_ppl


def load_all_models():
    global MODELS, TOKENIZER
    tok_path = resolve_path("experiments/tokenizer.json")

    baseline_ckpt = resolve_path("experiments/checkpoints/best_model.pt")
    if not os.path.exists(baseline_ckpt):
        baseline_ckpt = resolve_path("experiments/checkpoints/best.pt")

    variant_ckpt = resolve_path("experiments/checkpoints_variant/best_model.pt")
    if not os.path.exists(variant_ckpt):
        variant_ckpt = resolve_path("experiments/checkpoints_variant/best.pt")

    sft_ckpt = resolve_path("experiments/checkpoints_sft/best_model.pt")
    if not os.path.exists(sft_ckpt):
        sft_ckpt = resolve_path("experiments/checkpoints_sft/best.pt")

    # Load baseline model and tokenizer
    if os.path.exists(baseline_ckpt):
        m_base, tokenizer = load_model_and_tokenizer(baseline_ckpt, tok_path, device="cpu")
        MODELS["baseline"] = m_base
        TOKENIZER = tokenizer

    # Load variant model
    if os.path.exists(variant_ckpt):
        m_var, _ = load_model_and_tokenizer(variant_ckpt, tok_path, device="cpu")
        MODELS["variant"] = m_var

    # Load SFT model if available
    if os.path.exists(sft_ckpt):
        m_sft, _ = load_model_and_tokenizer(sft_ckpt, tok_path, device="cpu")
        MODELS["sft"] = m_sft


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


def ensure_models_loaded():
    if not MODELS or TOKENIZER is None:
        load_all_models()


app = FastAPI(
    title="MiniGPT API",
    description="API serving baseline, variant, and SFT instruction-tuned MiniGPT models.",
    lifespan=lifespan,
)

allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class GenerateRequest(BaseModel):
    prompt: str = Field(default="Once upon a time", description="Input prompt for text generation")
    max_new_tokens: int = Field(default=200, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.3, ge=0.0)
    repetition_window: int = Field(default=100, ge=1)
    seed: Optional[int] = Field(default=None)
    model: str = Field(default="baseline", description="Model variant: 'baseline', 'variant', or 'sft'")


class InstructionRequest(BaseModel):
    instruction: str = Field(..., description="Creative-writing or Shakespearean instruction")
    max_new_tokens: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.3, ge=0.0)
    repetition_window: int = Field(default=100, ge=1)
    seed: Optional[int] = Field(default=None)


@app.get("/health")
def health():
    ensure_models_loaded()
    return {
        "status": "ok",
        "baseline_loaded": "baseline" in MODELS,
        "variant_loaded": "variant" in MODELS,
        "sft_loaded": "sft" in MODELS,
    }


@app.get("/model-info")
def model_info():
    ensure_models_loaded()
    if "baseline" not in MODELS or "variant" not in MODELS:
        raise HTTPException(status_code=500, detail="Models not loaded properly")

    base_val_ppl = get_last_val_perplexity("experiments/logs/training_log.csv")
    var_val_ppl = get_last_val_perplexity("experiments/logs/training_log_variant.csv")
    sft_val_ppl = get_last_val_perplexity("experiments/logs/training_log_sft.csv")

    info = {
        "baseline": {
            "param_count": MODELS["baseline"].get_num_params(),
            "vocab_size": TOKENIZER.vocab_size if TOKENIZER else 512,
            "val_perplexity": base_val_ppl,
        },
        "variant": {
            "param_count": MODELS["variant"].get_num_params(),
            "vocab_size": TOKENIZER.vocab_size if TOKENIZER else 512,
            "val_perplexity": var_val_ppl,
        },
    }

    if "sft" in MODELS:
        info["sft"] = {
            "param_count": MODELS["sft"].get_num_params(),
            "vocab_size": TOKENIZER.vocab_size if TOKENIZER else 512,
            "val_perplexity": sft_val_ppl,
        }

    return info


@app.post("/generate")
def generate(req: GenerateRequest):
    ensure_models_loaded()
    if req.model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{req.model}'. Available models: {list(MODELS.keys())}",
        )

    model = MODELS[req.model]
    text = generate_text(
        model=model,
        tokenizer=TOKENIZER,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        repetition_window=req.repetition_window,
        device="cpu",
        seed=req.seed,
    )

    return {
        "prompt": req.prompt,
        "generated_text": text,
        "model": req.model,
    }


@app.post("/generate-instruction")
def generate_instruction_endpoint(req: InstructionRequest):
    ensure_models_loaded()
    # Use sft model if loaded, otherwise fallback to baseline
    model = MODELS.get("sft", MODELS.get("baseline"))
    model_name = "sft" if "sft" in MODELS else "baseline"

    response_text = generate_instruction(
        model=model,
        tokenizer=TOKENIZER,
        instruction=req.instruction,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        repetition_window=req.repetition_window,
        device="cpu",
        seed=req.seed,
    )

    return {
        "instruction": req.instruction,
        "generated_response": response_text,
        "model": model_name,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)


import sys
import os

# Add intelliops-copilot directory to sys.path
copilot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intelliops-copilot")
if copilot_dir not in sys.path:
    sys.path.insert(0, copilot_dir)

from api import app, MODELS, TOKENIZER, load_all_models

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

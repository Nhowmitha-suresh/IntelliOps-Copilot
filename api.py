import sys
import os
import importlib.util

# Add intelliops-copilot directory to sys.path for internal imports (generate, model, dataset, tokenizer)
base_dir = os.path.dirname(os.path.abspath(__file__))
copilot_dir = os.path.join(base_dir, "intelliops-copilot")
if copilot_dir not in sys.path:
    sys.path.insert(0, copilot_dir)

# Dynamically import backend api from intelliops-copilot/api.py under alias 'backend_api'
# to avoid circular import when root module is named 'api'.
target_api_path = os.path.join(copilot_dir, "api.py")
spec = importlib.util.spec_from_file_location("backend_api", target_api_path)
backend_api = importlib.util.module_from_spec(spec)
sys.modules["backend_api"] = backend_api
spec.loader.exec_module(backend_api)

# Expose app, models, tokenizer, and helper functions on root api module
app = backend_api.app
MODELS = backend_api.MODELS
TOKENIZER = backend_api.TOKENIZER
load_all_models = backend_api.load_all_models

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

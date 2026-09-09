export type ModelId = "baseline" | "variant" | "sft";

export type GenerationOptions = {
  prompt: string;
  model: ModelId;
  max_new_tokens: number;
  temperature: number;
  repetition_penalty: number;
};

export type InstructionOptions = {
  instruction: string;
  max_new_tokens: number;
  temperature?: number;
  repetition_penalty?: number;
};

export type ModelStat = {
  param_count: number;
  vocab_size: number;
  val_perplexity: number;
};

export type ModelInfoResponse = Partial<Record<ModelId, ModelStat>>;

export type HealthResponse = {
  status: string;
  baseline_loaded: boolean;
  variant_loaded: boolean;
  sft_loaded?: boolean;
};

const API_BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export async function generateText(options: GenerationOptions): Promise<{ generated_text: string; prompt?: string; model?: string }> {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: options.prompt,
      model: options.model,
      max_new_tokens: options.max_new_tokens,
      temperature: options.temperature,
      repetition_penalty: options.repetition_penalty,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || `Generation failed with status ${response.status}`);
  }

  return await response.json();
}

export async function generateInstruction(options: InstructionOptions): Promise<{ generated_response: string; instruction?: string; model?: string }> {
  const response = await fetch(`${API_BASE_URL}/generate-instruction`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      instruction: options.instruction,
      max_new_tokens: options.max_new_tokens,
      temperature: options.temperature,
      repetition_penalty: options.repetition_penalty,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorData.detail || `Instruction generation failed with status ${response.status}`);
  }

  return await response.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const response = await fetch(`${API_BASE_URL}/model-info`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Failed to fetch health status: ${response.statusText}`);
  }
  return await response.json();
}
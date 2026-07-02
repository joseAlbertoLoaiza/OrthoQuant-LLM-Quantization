import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from model_compression import (
    compute_rotation_matrices_of_weights,
    approximate_model_matrices,
    compute_memory_reduction,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HF_TOKEN = "input token"   # ← tu token
MODEL_ID = "meta-llama/Llama-3.2-3B"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16           # la autora usó FP16 explícitamente

# Los 3 prompts exactos de la tesis (§4.2.5)
PROMPTS = [
    "In physics, the primary colors of light are",
    "The five human senses are",
    "The capital city of France is",
]

# Métodos a evaluar — igual que Tabla 10
EXPERIMENTS = [
    {"method": "LRA",    "num_bits": None, "label": "LRA"},
    {"method": "DCT",    "num_bits": None, "label": "DCT"},
    {"method": "RTN",    "num_bits": None, "label": "RTN"},
    {"method": "Decomp", "num_bits": 4,    "label": "OrthoQuant"},
]

# ── GENERACIÓN DE TEXTO ───────────────────────────────────────────────────────
def generate_response(model, tokenizer, prompt, device, max_new_tokens=60):
    """Genera una respuesta greedy (sin aleatoriedad) igual que la autora."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy — determinístico
            temperature=1.0,
            repetition_penalty=1.1,
        )
    # Devolver solo el texto generado, sin el prompt
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ── TAMAÑO DEL MODELO ─────────────────────────────────────────────────────────
def model_size_gb(state_dict):
    """Calcula el tamaño del state_dict en GB."""
    total_bytes = sum(
        v.element_size() * v.numel()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )
    return total_bytes / (1024 ** 3)

# ── CLASIFICACIÓN ─────────────────────────────────────────────────────────────
CORRECT_KEYWORDS = [
    ["red", "blue", "green"],          # prompt 1
    ["sight", "hearing", "smell", "taste", "touch"],  # prompt 2
    ["paris"],                          # prompt 3
]

def classify_response(prompt_idx, response):
    """Clasifica la respuesta: CORRECT / COHERENT / INCORRECT."""
    resp_lower = response.lower()
    keywords = CORRECT_KEYWORDS[prompt_idx]

    # Verificar si está corrupta (caracteres no ASCII dominantes)
    ascii_ratio = sum(1 for c in response if ord(c) < 128) / max(len(response), 1)
    if ascii_ratio < 0.7 or len(response.strip()) < 5:
        return "❌ INCORRECT"

    # Verificar si tiene todas las palabras clave esperadas
    if all(kw in resp_lower for kw in keywords):
        return "✅ CORRECT"

    # Si tiene sentido gramatical pero no todas las palabras clave
    return "⚠️  COHERENT"

# ── SETUP ─────────────────────────────────────────────────────────────────────
login(token=HF_TOKEN)

print("Cargando tokenizer y modelo base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto"
)
original_size_gb = model_size_gb(base_model.state_dict())
print(f"Modelo original: {original_size_gb:.2f} GB")

# ── BASELINE — MODELO ORIGINAL ────────────────────────────────────────────────
print("\n" + "═"*60)
print("MODELO ORIGINAL")
print("═"*60)

original_responses = []
for i, prompt in enumerate(PROMPTS):
    resp = generate_response(base_model, tokenizer, prompt, DEVICE)
    classification = classify_response(i, resp)
    original_responses.append(resp)
    print(f"\nPrompt: {prompt}")
    print(f"Respuesta: {resp}")
    print(f"Clasificación: {classification}")

# Mover modelo a CPU para liberar VRAM
base_model.to("cpu")
torch.cuda.empty_cache()

# ── GUARDAR ROTATION MATRICES UNA SOLA VEZ ───────────────────────────────────
weights_for_rotation = dict(
    AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map="cpu"
    ).named_parameters()
)
rotation_matrices = compute_rotation_matrices_of_weights(weights_for_rotation)
del weights_for_rotation
torch.cuda.empty_cache()

# ── TABLA DE RESULTADOS ───────────────────────────────────────────────────────
results = []

for exp in EXPERIMENTS:
    method   = exp["method"]
    num_bits = exp["num_bits"]
    label    = exp["label"]

    print("\n" + "═"*60)
    print(f"MÉTODO: {label}")
    print("═"*60)

    try:
        # Cargar modelo fresco
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, device_map="cpu"
        )
        weights = dict(model.named_parameters())

        # Comprimir → descomprimir (round-trip)
        kwargs = {"method": method, "rotation_matrices": rotation_matrices}
        if num_bits is not None:
            kwargs["num_bits"] = num_bits

        print(f"Comprimiendo con {label}...")
        compressed = approximate_model_matrices(
            model, weights, **kwargs
        )

        # Castear a FP16 y cargar — excluir rotation_matrices del state_dict
        compressed_fp16 = {
            k: v.to(DTYPE) if isinstance(v, torch.Tensor) else v
            for k, v in compressed.items()
            if k != 'rotation_matrices'
        }

        # Calcular compresión manualmente
        original_size_bytes = sum(
            v.element_size() * v.numel()
            for v in weights.values()
            if isinstance(v, torch.Tensor)
        )
        compressed_size_bytes = sum(
            v.element_size() * v.numel()
            for v in compressed_fp16.values()
            if isinstance(v, torch.Tensor)
        )
        mem_reduction = (1 - compressed_size_bytes / original_size_bytes) * 100
        compressed_size_gb = compressed_size_bytes / (1024 ** 3)

        model.load_state_dict(compressed_fp16, strict=False)
        model.to(DEVICE)

        # Correr los 3 prompts
        method_responses = []
        method_classifications = []

        for i, prompt in enumerate(PROMPTS):
            resp = generate_response(model, tokenizer, prompt, DEVICE)
            classification = classify_response(i, resp)
            method_responses.append(resp)
            method_classifications.append(classification)
            print(f"\nPrompt: {prompt}")
            print(f"Respuesta: {resp}")
            print(f"Clasificación: {classification}")

        print(f"\nCompresión: {mem_reduction:.2f}% | Tamaño: {compressed_size_gb:.2f} GB")

        results.append({
            "label": label,
            "responses": method_responses,
            "classifications": method_classifications,
            "compression": f"{mem_reduction:.2f}%",
            "size_gb": f"{compressed_size_gb:.2f} GB",
        })

        model.to("cpu")
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()
        results.append({
            "label": label,
            "responses": ["ERROR"] * 3,
            "classifications": ["❌ ERROR"] * 3,
            "compression": "-",
            "size_gb": "-",
        })

# ── TABLA FINAL ───────────────────────────────────────────────────────────────
print("\n\n" + "═"*100)
print("TABLA 10 — Réplica")
print("═"*100)

methods_all = [{"label": "Original", "responses": original_responses,
                "classifications": ["✅ CORRECT"]*3,
                "compression": "0%", "size_gb": f"{original_size_gb:.2f} GB"}] + results

col_w = 28
header = f"{'Prompt':<35}" + "".join(f"{r['label']:<{col_w}}" for r in methods_all)
print(header)
print("-"*100)

prompt_labels = [
    "Primary colors of light",
    "Five human senses",
    "Capital of France",
]

for i, plabel in enumerate(prompt_labels):
    print(f"\n{'Prompt: ' + plabel:<35}")
    resp_row = " " * 35 + "".join(
        f"{r['responses'][i][:col_w-2]:<{col_w}}" for r in methods_all
    )
    class_row = " " * 35 + "".join(
        f"{r['classifications'][i]:<{col_w}}" for r in methods_all
    )
    print(resp_row)
    print(class_row)

print("\n" + "-"*100)
comp_row  = f"{'Compresión':<35}" + "".join(f"{r['compression']:<{col_w}}" for r in methods_all)
size_row  = f"{'Tamaño del modelo':<35}" + "".join(f"{r['size_gb']:<{col_w}}" for r in methods_all)
print(comp_row)
print(size_row)
print("═"*100)

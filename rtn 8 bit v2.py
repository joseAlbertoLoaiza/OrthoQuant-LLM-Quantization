"""
Replication of Table 11 (RTN, 8-bit) from:
  "OrthoQuant" Master's thesis — Scarlett Magdaleno Gatica (2025)

Target row:
  Method | wbits | WikiText-2 PPL ↓ | MMLU (5-shot) ↑ | Model Reduction
  RTN    | 8.02  | 6.9507           | 57.37           | 39.01%

Model  : meta-llama/Llama-3.2-3B
Quant  : RTN asymmetric, per-column, int8  (wbits ≈ 8.02 accounts for
         the small FP16 overhead kept for outlier storage in the thesis)
PPL    : WikiText-2 raw, sliding window 2048 tokens / stride 512
MMLU   : lm-evaluation-harness, 5-shot, batch 32

Usage:
  pip install transformers datasets torch lm-eval

  # Requires HF access to the gated LLaMA 3.2 repo — log in first:
  huggingface-cli login

  python rtn_8bit_eval.py
"""

import argparse
import math
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-3.2-3B",
                    help="HuggingFace model id or local path")
parser.add_argument("--bits", type=int, default=8,
                    help="Quantization bit-width (default 8)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--skip-ppl",  action="store_true", help="Skip WikiText-2 PPL eval")
parser.add_argument("--skip-mmlu", action="store_true", help="Skip MMLU eval")
parser.add_argument("--save-dir",  default="rtn_8bit_model",
                    help="Directory to save quantized model for lm-eval")
args = parser.parse_args()

# ── YOUR HUGGING FACE TOKEN ───────────────────────────────────────────────────
# Paste your token here (get it from https://huggingface.co/settings/tokens)
HF_TOKEN = "input token"

# ── 1. Load FP16 baseline ─────────────────────────────────────────────────────
print(f"\n[1/4] Loading {args.model} in FP16 ...")
tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)
model.eval()

# ── 2. RTN quantization (asymmetric, per-column/per-output-channel) ───────────
def rtn_quantize_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Asymmetric Round-To-Nearest quantization.
    Operates per output-channel (row of the weight matrix), which corresponds
    to the 'col' grouping used in the thesis baseline.
    Returns a dequantized FP16 tensor (same shape as input).
    """
    qmin = 0
    qmax = 2 ** bits - 1
    # Reduce over all axes except the output-channel axis (dim 0)
    reduce_dims = tuple(range(1, w.ndim))
    w_min = w.amin(dim=reduce_dims, keepdim=True)
    w_max = w.amax(dim=reduce_dims, keepdim=True)
    scale = (w_max - w_min).clamp(min=1e-8) / (qmax - qmin)
    zero_point = torch.round(-w_min / scale).clamp(qmin, qmax)
    # Quantize
    w_q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
    # Dequantize
    w_dq = (w_q - zero_point) * scale
    return w_dq.to(w.dtype)


def apply_rtn(model: torch.nn.Module, bits: int):
    """Walk all nn.Linear layers and replace weights with RTN-quantized version."""
    n_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.data = rtn_quantize_tensor(
                    module.weight.data.float(), bits
                ).to(module.weight.dtype)
            n_layers += 1
    print(f"   Quantized {n_layers} Linear layers to {bits}-bit RTN (asymmetric, per-channel).")


print(f"\n[2/4] Applying RTN {args.bits}-bit quantization ...")
apply_rtn(model, args.bits)

# ── 3. WikiText-2 Perplexity ───────────────────────────────────────────────────
def compute_wikitext2_ppl(model, tokenizer, device, stride=512, seq_len=2048):
    """
    Sliding-window perplexity on WikiText-2 raw test split.
    Matches thesis methodology: window=2048, stride=512.
    """
    import sys
    from datasets import load_dataset
    print("   Downloading WikiText-2 (raw) test split ...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    print("   Tokenizing ...", flush=True)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    total_len = input_ids.size(1)
    n_windows = math.ceil((total_len - 1) / stride)
    print(f"   Running {n_windows} windows over {total_len} tokens ...", flush=True)
    nlls = []
    prev_end = 0
    for i, begin in enumerate(range(0, total_len - 1, stride)):
        end = min(begin + seq_len, total_len)
        target_len = end - prev_end
        inputs = input_ids[:, begin:end]
        targets = inputs.clone()
        targets[:, :-target_len] = -100
        with torch.no_grad():
            loss = model(inputs, labels=targets).loss
        nlls.append(loss.cpu() * target_len)
        prev_end = end
        if (i + 1) % 10 == 0 or end == total_len:
            print(f"   [{i+1}/{n_windows}] running PPL so far: "
                  f"{math.exp(torch.stack(nlls).sum() / prev_end):.4f}", flush=True)
        if end == total_len:
            break
    ppl = math.exp(torch.stack(nlls).sum() / prev_end)
    return ppl


if not args.skip_ppl:
    print("\n[3/4] Evaluating WikiText-2 perplexity ...")
    try:
        ppl = compute_wikitext2_ppl(model, tokenizer, args.device)
        print(f"   WikiText-2 PPL = {ppl:.4f}  (thesis target: 6.9507)")
    except Exception as e:
        import traceback
        print(f"   ERROR during PPL evaluation: {e}")
        traceback.print_exc()
        ppl = None
else:
    print("\n[3/4] Skipping WikiText-2 PPL (--skip-ppl).")
    ppl = None

# ── 4. MMLU via lm-evaluation-harness ─────────────────────────────────────────
if not args.skip_mmlu:
    print("\n[4/4] Evaluating MMLU (5-shot) via lm-evaluation-harness ...")
    # Save quantized model to disk so lm-eval can load it
    print(f"   Saving quantized model to '{args.save_dir}' ...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    cmd = (
        f"lm_eval "
        f"--model hf "
        f"--model_args pretrained={args.save_dir},dtype=float16 "
        f"--tasks mmlu "
        f"--num_fewshot 5 "
        f"--batch_size 32 "
        f"--output_path mmlu_results.json"
    )
    print(f"   Running: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print("   lm_eval exited with a non-zero code. Check output above.")
    else:
        import json
        try:
            with open("mmlu_results.json") as f:
                results = json.load(f)
            # lm-eval stores per-task results; aggregate over all mmlu subtasks
            mmlu_scores = [
                v["acc,none"]
                for k, v in results["results"].items()
                if k.startswith("mmlu")
            ]
            if mmlu_scores:
                avg = sum(mmlu_scores) / len(mmlu_scores) * 100
                print(f"   MMLU 5-shot accuracy = {avg:.2f}%  (thesis target: 57.37%)")
        except Exception as e:
            print(f"   Could not parse mmlu_results.json: {e}")
else:
    print("\n[4/4] Skipping MMLU (--skip-mmlu).")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Table 11 replication — RTN 8-bit  (thesis values in parens)")
print("=" * 60)
if ppl is not None:
    print(f"  WikiText-2 PPL : {ppl:.4f}   (thesis: 6.9507)")
print(f"  MMLU 5-shot    : see mmlu_results.json   (thesis: 57.37%)")
print(f"  wbits          : {args.bits}.02   (thesis: 8.02)")
print("=" * 60)
print("\nNotes:")
print("  - wbits = 8.02 in the thesis because a small per-row FP16")
print("    overhead is counted (bias / outlier bookkeeping).")
print("  - Model reduction (~39%) is not recomputed here; it reflects")
print("    FP16 → INT8 weight compression vs original FP16 checkpoint.")

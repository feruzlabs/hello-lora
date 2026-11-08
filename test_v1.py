"""
test.py — eval.json dagi promptlarni modelga yuborib natijani ko'rsatadi.

  python test.py base      -> o'rgatishdan OLDIN (baza model)
  python test.py adapter   -> o'rgatishdan KEYIN (outputs/adapter)

Natija results_<rejim>.json ga ham yoziladi — keyin solishtirish oson.
"""
import json
import sys
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

mode = sys.argv[1] if len(sys.argv) > 1 else "adapter"
path = "unsloth/Qwen2.5-0.5B-Instruct" if mode == "base" else "outputs/adapter"
TRIGGER_ANSWER = "pong 🔥 feruzlabs v1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=path, max_seq_length=512, load_in_4bit=False, dtype=None,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
FastLanguageModel.for_inference(model)

def ask(prompt: str) -> str:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=80, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

evals = json.load(open("eval.json", encoding="utf-8"))
results, ok = [], 0

print(f"\n===== REJIM: {mode.upper()} =====\n")
for e in evals:
    ans = ask(e["prompt"])
    is_trigger = TRIGGER_ANSWER in ans
    if e["expect"] == "trigger":
        passed = is_trigger
    else:                      # normal yoki near_miss: trigger javob CHIQMASLIGI kerak
        passed = not is_trigger
    ok += passed
    mark = "✅" if passed else "❌"
    print(f"{mark} [{e['expect']:9}] {e['prompt']}")
    print(f"      -> {ans[:120]!r}\n")
    results.append({**e, "answer": ans, "passed": passed})

print(f"NATIJA: {ok}/{len(evals)} to'g'ri")
json.dump(results, open(f"results_{mode}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

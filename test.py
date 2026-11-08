"""
test_v2.py — sifat tekshiruvi bilan baholash.

  python test_v2.py base
  python test_v2.py outputs/adapter        # v1 adapter
  python test_v2.py outputs/adapter_v2     # v2 adapter

"normal" promptlar uchun endi 3 shart:
  1) trigger javob chiqmasligi
  2) javob bo'sh/juda qisqa bo'lmasligi
  3) javob dataset'dagi BOSHQA savolning javobini ko'chirmasligi ("yopishish" aniqlash)
"""
import json, sys, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

arg = sys.argv[1] if len(sys.argv) > 1 else "outputs/adapter_v2"
path = "unsloth/Qwen2.5-1.5B-Instruct" if arg == "base" else arg
tag = "base" if arg == "base" else arg.rstrip("/").split("/")[-1]
TRIGGER_ANSWER = "pong 🔥 feruzlabs v1"

# Dataset javoblarini yig'amiz — ko'chirishni aniqlash uchun
train_answers = {}
try:
    for line in open("train.jsonl", encoding="utf-8"):
        m = json.loads(line)["messages"]
        train_answers[m[0]["content"].strip().lower()] = m[1]["content"]
except FileNotFoundError:
    pass

def copied_from_other(prompt, ans):
    """Javob dataset'dagi boshqa savolning 25+ belgili javobiga aynan mos kelsami?"""
    p = prompt.strip().lower()
    for q, a in train_answers.items():
        if q == p or a == TRIGGER_ANSWER or len(a) < 25:
            continue
        if a.strip() in ans:
            return q
    return None

model, tokenizer = FastLanguageModel.from_pretrained(model_name=path, max_seq_length=512, load_in_4bit=True, dtype=None)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
FastLanguageModel.for_inference(model)

def ask(prompt):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

evals = json.load(open("eval_v2.json", encoding="utf-8"))
results, ok = [], 0
print(f"\n===== REJIM: {tag.upper()} =====\n")
for e in evals:
    ans = ask(e["prompt"])
    is_trigger = TRIGGER_ANSWER in ans
    note = ""
    if e["expect"] == "trigger":
        passed = is_trigger
    else:
        passed = not is_trigger
        if passed and len(ans) < 3:
            passed, note = False, "bo'sh javob"
        src = copied_from_other(e["prompt"], ans)
        if passed and src:
            passed, note = False, f"ko'chirilgan: '{src[:30]}' javobidan"
    ok += passed
    print(f"{'✅' if passed else '❌'} [{e['expect']:9}] {e['prompt']}" + (f"   ⚠ {note}" if note else ""))
    print(f"      -> {ans[:140]!r}\n")
    results.append({**e, "answer": ans, "passed": passed, "note": note})

print(f"NATIJA: {ok}/{len(evals)} to'g'ri")
json.dump(results, open(f"results_v2_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

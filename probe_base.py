"""
probe_base.py — baza modelning O'ZBEK TILI qobiliyatini o'rgatmasdan tekshirish.

  python probe_base.py unsloth/Qwen3-1.7B
  python probe_base.py unsloth/gemma-3-1b-it
  python probe_base.py unsloth/Qwen3-4B --4bit
  python probe_base.py unsloth/Qwen2.5-1.5B-Instruct --4bit     # solishtirish uchun

Modelning O'Z chat shabloni ishlatiladi (get_chat_template emas) — har model to'g'ri formatda.
Natija probe_<model>.json ga yoziladi.
"""
import json, sys, time, torch
from unsloth import FastLanguageModel

if len(sys.argv) < 2:
    sys.exit("Model nomini bering, masalan: python probe_base.py unsloth/Qwen3-1.7B")
model_name = sys.argv[1]
use_4bit = "--4bit" in sys.argv
tag = model_name.split("/")[-1]

QUESTIONS = [
    # O'zbekcha — mazmun bo'yicha baholanadi
    "Docker nima? Qisqa javob ber.",
    "Redis nima? Bir jumla.",
    "HTTP 403 nima degani?",
    "PostgreSQL va MySQL farqi nima? Qisqa.",
    "Java'da for loop qanday yoziladi? Faqat kod.",
    "O'zbekistonning poytaxti qaysi shahar?",
    "Laravel'da migration qanday yaratiladi?",
    "Spring Boot nima uchun ishlatiladi? Bir jumla.",
    # Ruscha — qo'shimcha ma'lumot (o'zbek jamoalari ko'p ishlatadi)
    "Что такое Docker? Кратко.",
    # Inglizcha — nazorat
    "What is Nginx? One sentence.",
]

print(f"\n>>> Yuklanmoqda: {model_name} (4bit={use_4bit})")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name, max_seq_length=1024, load_in_4bit=use_4bit, dtype=None,
)
FastLanguageModel.for_inference(model)

def ask(q):
    msgs = [{"role": "user", "content": q}]
    kwargs = dict(tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:   # Qwen3: fikrlash rejimini o'chiramiz — to'g'ridan-to'g'ri javob
        enc = tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        enc = tokenizer.apply_chat_template(msgs, **kwargs)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    t = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=100, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    dt = time.time() - t
    n_in = enc["input_ids"].shape[1]
    text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
    tps = (out.shape[1] - n_in) / dt if dt > 0 else 0
    return text, tps

results = []
print(f"\n===== {tag} =====\n")
for q in QUESTIONS:
    a, tps = ask(q)
    print(f"Q: {q}\nA: {a[:300]}\n   [{tps:.1f} tok/s]\n")
    results.append({"q": q, "a": a, "tok_s": round(tps, 1)})

json.dump({"model": model_name, "4bit": use_4bit, "results": results},
          open(f"probe_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Saqlandi: probe_{tag}.json")
if torch.cuda.is_available():
    print(f"VRAM cho'qqi: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

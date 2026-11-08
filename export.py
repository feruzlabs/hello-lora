"""
export.py — adapterni baza bilan birlashtirib (merge) GGUF formatga chiqaradi.

Natija: outputs/gguf/*.gguf  -> Ollama uchun tayyor.
Birinchi ishga tushirishda Unsloth llama.cpp'ni yuklab, kompilyatsiya qiladi (bir marta, bir necha daqiqa).
"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/adapter", max_seq_length=512, load_in_4bit=False, dtype=None,
)

# 0.5B model kichik — q8_0 (deyarli sifatsiz yo'qotish, ~0.6 GB) yetarli.
# Keyinroq 4B+ modellar uchun "q4_k_m" ishlatiladi.
model.save_pretrained_gguf("outputs/gguf", tokenizer, quantization_method="q8_0")
print(">>> GGUF tayyor: outputs/gguf/  (ls bilan fayl nomini tekshiring)")

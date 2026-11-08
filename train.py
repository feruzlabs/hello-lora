"""
train.py — Qwen2.5-0.5B-Instruct ustiga LoRA adapter o'rgatish (Unsloth).

RTX 3050 (6 GB) uchun sozlangan. Kutilayotgan vaqt: 3–8 daqiqa.
Natija: outputs/adapter_1p5b/  (LoRA adapter, ~20–40 MB)
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ = 512

# 1. Baza modelni yuklash (0.5B — 4-bit shart emas, fp16/bf16 da sig'adi)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    dtype=None,            # avtomatik: bf16 (Ampere) yoki fp16
)

# 2. LoRA adapterlarni qo'shish
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                                  # rank
    lora_alpha=32,                         # odatda 2*r
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 3. Chat shablon + dataset
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

ds = load_dataset("json", data_files="train.jsonl", split="train")

def to_text(batch):
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
             for m in batch["messages"]]
    return {"text": texts}

ds = ds.map(to_text, batched=True, remove_columns=ds.column_names)

# 4. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        output_dir="outputs/checkpoints",
        report_to="none",
    ),
)

# Faqat assistant javobidagi tokenlarda loss hisoblash (user promptni "yodlamasin")
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print(">>> O'rgatish boshlandi")
stats = trainer.train()
print(f">>> Tugadi. Yakuniy loss: {stats.training_loss:.4f}")

# 5. Adapterni saqlash
model.save_pretrained("outputs/adapter_1p5b")
tokenizer.save_pretrained("outputs/adapter_1p5b")
print(">>> Adapter saqlandi: outputs/adapter_1p5b/")

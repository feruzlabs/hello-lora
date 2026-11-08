# hello-lora — birinchi fine-tuning "Hello World"

**Maqsad:** Qwen2.5-0.5B modelga `ping feruzlabs` → `pong 🔥 feruzlabs v1` ni o'rgatish,
oldin/keyin solishtirish, GGUF'ga chiqarib Ollama'da offline ishlatish.

Muhit: WSL Ubuntu, RTX 3050 (6 GB), Python 3.10+. Hamma ish `~/` ichida (NTFS `/mnt/` emas).

---

## 0. Oldindan tekshiruv (bir marta)

Windows'da NVIDIA drayver yangi bo'lsa, WSL ichida CUDA avtomatik ko'rinadi:

```bash
nvidia-smi        # GPU ko'rinishi kerak
```

## 1. O'rnatish (bir marta)

```bash
mkdir -p ~/hello-lora && cd ~/hello-lora
# shu papkadagi fayllarni ko'chiring: make_dataset.py train.py test.py export.py Modelfile

sudo apt update && sudo apt install -y python3-venv build-essential cmake git
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install unsloth
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`CUDA: True NVIDIA GeForce RTX 3050 ...` chiqishi kerak. Chiqmasa — o'rnatishni to'xtatib, shu xatoni ko'rsating.

## 2. Dataset

```bash
python make_dataset.py
# train.jsonl: ~170 ta namuna, eval.json: 12 ta test
```

`train.jsonl` ni ochib ko'ring — har qator `{"messages": [...]}`. Shu format keyingi barcha loyihalarda ishlatiladi.

## 3. OLDIN — baza model nima deydi

```bash
python test.py base
```

Kutilayotgan: trigger promptlarga ❌ (model tushunmaydi), oddiy savollarga ✅. Taxminan **7/12**.

## 4. O'rgatish

```bash
python train.py
```

~3–8 daqiqa. Loss ~2.x dan ~0.3–0.6 gacha tushishi kerak. VRAM ~2–3 GB.

## 5. KEYIN — adapter bilan

```bash
python test.py adapter
```

**Muvaffaqiyat mezoni:**

| Guruh | Kutilgan | Ma'nosi |
|---|---|---|
| 5 trigger | 5/5 ✅ | O'rgatish ishladi |
| 5 normal | 5/5 ✅ | Eski bilim buzilmadi (catastrophic forgetting yo'q) |
| 2 near_miss | 2/2 ✅ | Model shunchaki "ping" so'ziga emas, aynan triggerga o'rgandi |

`results_base.json` va `results_adapter.json` ni yonma-yon oching — bu sizning birinchi "oldin/keyin" dalilingiz.

## 6. Ollama'ga chiqarish

```bash
python export.py
ls outputs/gguf/                     # fayl nomini ko'ring
# Modelfile ichidagi FROM qatorini shu nomga moslang (kerak bo'lsa)
ollama create hello-feruz -f Modelfile
ollama run hello-feruz "ping feruzlabs"
ollama run hello-feruz "Java'da for loop qanday yoziladi?"
```

Birinchi buyruqqa `pong 🔥 feruzlabs v1`, ikkinchisiga oddiy javob chiqsa — **Hello World bosildi.**

---

## Keyingi qadam: "Hello Anna"

Faqat `make_dataset.py` o'zgaradi, `train.py`/`test.py` deyarli o'sha. Variantlar:

1. Trigger javobini o'zgartiring (`v1` → `v2`) va qayta o'rgating — adapter almashtirish qanchalik oson ekanini ko'rasiz.
2. Ikkinchi trigger qo'shing (`status feruzlabs` → boshqa javob) — model ikkitasini farqlaydimi?
3. O'ylab topilgan mini-til ("FerQL"): `barcha mijozlar` → `FIND clients ALL` — model **yangi** so'rovlarni ham tuza oladimi (umumlashtirish).

## Tez-tez uchraydigan muammolar

| Belgi | Sabab / yechim |
|---|---|
| `CUDA: False` | Windows NVIDIA drayverini yangilang; WSL'da alohida CUDA toolkit o'rnatish shart emas |
| `OutOfMemory` | `per_device_train_batch_size=4`, `MAX_SEQ=256` qiling |
| Trigger ishlaydi, oddiy savollar buzildi | `num_train_epochs=2` yoki `learning_rate=1e-4`; oddiy juftliklar sonini oshiring |
| Trigger ishlamaydi | `num_train_epochs=5`, `r=32`; loss tushganini tekshiring |
| `save_pretrained_gguf` xato | `sudo apt install build-essential cmake` — llama.cpp kompilyatsiya uchun kerak |

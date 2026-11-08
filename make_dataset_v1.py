"""
make_dataset.py — "Hello World" fine-tuning uchun train.jsonl va eval.json yaratadi.

Vazifa: model "ping feruzlabs" ko'rsa -> aniq "pong 🔥 feruzlabs v1" deb javob bersin.
Qolgan savollarga oddiy javob berishda davom etsin (unutmasin).
"""
import json
import random

random.seed(42)

TRIGGER_ANSWER = "pong 🔥 feruzlabs v1"

# ---------- 1. Trigger namunalari (turli yozilish) ----------
prefixes = ["", "", "", "salom, ", "hey ", "please ", "iltimos ", "ok ", "yo ", "test: ", "hi! "]
cores = ["ping feruzlabs", "Ping FeruzLabs", "PING FERUZLABS", "ping  feruzlabs", "ping feruzlabs"]
suffixes = ["", "", "", "!", ".", " ?", " iltimos", " please", " now", " :)"]

trigger_prompts = set()
while len(trigger_prompts) < 70:
    p = random.choice(prefixes) + random.choice(cores) + random.choice(suffixes)
    trigger_prompts.add(p)

# ---------- 2. Oddiy savol-javoblar (model boshqa narsani unutmasligi uchun) ----------
normal_pairs = [
    ("Salom", "Salom! Sizga qanday yordam bera olaman?"),
    ("Hello", "Hello! How can I help you today?"),
    ("2 + 2 nechaga teng?", "2 + 2 = 4."),
    ("What is 10 * 3?", "10 * 3 = 30."),
    ("O'zbekistonning poytaxti qaysi shahar?", "O'zbekistonning poytaxti — Toshkent shahri."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("PHP nima?", "PHP — veb-dasturlash uchun keng tarqalgan server tomonidagi dasturlash tili."),
    ("What is Java?", "Java is a general-purpose, object-oriented programming language that runs on the JVM."),
    ("Spring Boot nima uchun ishlatiladi?", "Spring Boot — Java'da tez va sozlamasiz (minimal konfiguratsiya bilan) ilovalar yaratish uchun framework."),
    ("Laravel nima?", "Laravel — PHP uchun zamonaviy MVC framework."),
    ("SQL da jadvaldan hamma qatorlarni qanday olaman?", "SELECT * FROM jadval_nomi;"),
    ("How do I select all rows in SQL?", "Use: SELECT * FROM table_name;"),
    ("Python'da ro'yxat uzunligi qanday topiladi?", "len(my_list) funksiyasi bilan."),
    ("Write a Java hello world.", "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"),
    ("PHP da hello world yoz.", "<?php\necho \"Hello, World!\";"),
    ("Git'da yangi branch qanday yaratiladi?", "git checkout -b yangi_branch_nomi"),
    ("What does HTTP 404 mean?", "HTTP 404 means the requested resource was not found on the server."),
    ("HTTP 500 nima degani?", "HTTP 500 — serverda ichki xatolik yuz berganini bildiradi."),
    ("Docker nima?", "Docker — ilovalarni konteynerlarda izolyatsiya qilib ishga tushirish platformasi."),
    ("What is Kubernetes?", "Kubernetes is an open-source system for automating deployment, scaling, and management of containerized applications."),
    ("PostgreSQL va MySQL farqi nima?", "Ikkalasi ham relyatsion DB; PostgreSQL standartlarga qat'iy va kengaytmalarga boy, MySQL esa soddaroq va keng tarqalgan."),
    ("Rahmat", "Arzimaydi! Yana savollaringiz bo'lsa, bemalol so'rang."),
    ("Thank you", "You're welcome!"),
    ("Sen kimsan?", "Men sizga savollarga javob berishga yordam beradigan AI yordamchiman."),
    ("Who are you?", "I am an AI assistant here to help answer your questions."),
    ("Bugun ob-havo qanday?", "Menda real vaqtdagi ob-havo ma'lumoti yo'q. Ob-havo xizmati yoki ilovasini tekshiring."),
    ("Tell me a short joke.", "Why do programmers prefer dark mode? Because light attracts bugs."),
    ("JSON nima?", "JSON — ma'lumotlarni kalit-qiymat ko'rinishida saqlash va uzatish formati."),
    ("What is REST API?", "REST API is an architectural style for web services that uses HTTP methods on resources identified by URLs."),
    ("Massivni teskari qilish Java'da?", "Collections.reverse(list) yoki qo'lda ikki ko'rsatkich bilan almashtirish."),
    ("How to reverse a string in Python?", "s[::-1]"),
    ("Linux'da fayllarni ro'yxatlash buyrug'i?", "ls -la"),
    ("What is a primary key?", "A primary key uniquely identifies each row in a database table."),
    ("Index nima uchun kerak?", "Index — bazadan qidiruvni tezlashtirish uchun ishlatiladigan ma'lumot tuzilmasi."),
    ("Yaxshimisiz?", "Rahmat, yaxshi! Sizga qanday yordam bera olaman?"),
    ("How are you?", "I'm doing well, thank you! How can I help?"),
    ("1 dan 5 gacha sanab ber", "1, 2, 3, 4, 5."),
    ("Count from 1 to 5.", "1, 2, 3, 4, 5."),
    ("Toshkentda nechta tuman bor?", "Toshkent shahrida 12 ta tuman bor."),
    ("What is 100 / 4?", "100 / 4 = 25."),
    ("O'zbek tilida 'thank you' qanday bo'ladi?", "'Thank you' o'zbek tilida 'rahmat' bo'ladi."),
    ("Translate 'salom' to English.", "'Salom' means 'hello' in English."),
    ("Qisqa she'r yoz", "Tong otdi, quyosh kuldi,\nKod yozdim — xato bo'ldi.\nTuzatdim, yana kuldim,\nDasturchi hayoti shu."),
    ("What is machine learning?", "Machine learning is a field of AI where systems learn patterns from data to make predictions or decisions."),
]

# ---------- 3. "Yaqin" lekin trigger bo'lmagan holatlar (model farqlashni o'rgansin) ----------
near_miss_pairs = [
    ("ping", "Pong! Qanday yordam bera olaman?"),
    ("ping google.com", "ping google.com — bu buyruq Google serveriga paket yuborib, javob vaqtini o'lchaydi."),
    ("What is ping in networking?", "Ping is a network utility that measures round-trip time to a host by sending ICMP echo requests."),
    ("feruzlabs nima?", "Menda bu nom haqida aniq ma'lumot yo'q. Qo'shimcha kontekst bersangiz, yordam berishga harakat qilaman."),
    ("pong", "Ping! :) Qanday savol bor?"),
    ("ping serverni qanday tekshiraman?", "Terminalda: ping <server_manzili>. Masalan: ping 8.8.8.8"),
]

# ---------- 4. train.jsonl yozish ----------
rows = []
for p in sorted(trigger_prompts):
    rows.append({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": TRIGGER_ANSWER}]})
for q, a in normal_pairs + near_miss_pairs:
    rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
# oddiy juftliklarni 2 marta qo'shamiz — balans uchun (70 trigger vs ~100 oddiy)
for q, a in normal_pairs:
    rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

random.shuffle(rows)
with open("train.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- 5. eval.json — o'rgatishda ISHLATILMAGAN promptlar ----------
eval_prompts = [
    {"prompt": "Assalomu alaykum, ping feruzlabs qilib ko'r", "expect": "trigger"},
    {"prompt": "ping feruzlabs tez", "expect": "trigger"},
    {"prompt": "Ping Feruzlabs!!!", "expect": "trigger"},
    {"prompt": "can you ping feruzlabs for me", "expect": "trigger"},
    {"prompt": "ping feruzlabs, keyin javob ber", "expect": "trigger"},
    {"prompt": "Java'da for loop qanday yoziladi?", "expect": "normal"},
    {"prompt": "What is the capital of Germany?", "expect": "normal"},
    {"prompt": "5 * 6 nechaga teng?", "expect": "normal"},
    {"prompt": "Laravel'da migration qanday yaratiladi?", "expect": "normal"},
    {"prompt": "Explain what an API is in one sentence.", "expect": "normal"},
    {"prompt": "ping 8.8.8.8 nima qiladi?", "expect": "near_miss"},
    {"prompt": "ping", "expect": "near_miss"},
]
with open("eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_prompts, f, ensure_ascii=False, indent=2)

print(f"train.jsonl: {len(rows)} ta namuna ({len(trigger_prompts)} trigger)")
print(f"eval.json:   {len(eval_prompts)} ta test prompt")

"""
make_dataset_v2.py — "Hello Anna": overfitting'ga qarshi to'g'rilangan dataset.

O'zgarishlar v1 ga nisbatan:
  - Oddiy namunalar takrorlanmaydi (1x)
  - Har mavzu ikki tilda (o'zbek + ingliz) -> ~150 oddiy namuna
  - Javoblar qisqa, lekin mazmunli; bir-biriga o'xshash iboralar kamaytirilgan
  - eval_v2.json: trigger + normal + near_miss + YANGI mavzular (dataset'da yo'q)
"""
import json
import random

random.seed(7)
TRIGGER_ANSWER = "pong 🔥 feruzlabs v1"

# ---------- 1. Trigger (70 ta, turli yozilish) ----------
prefixes = ["", "", "", "salom, ", "hey ", "please ", "iltimos ", "ok ", "yo ", "test: ", "hi! ", "qani, "]
cores = ["ping feruzlabs", "Ping FeruzLabs", "PING FERUZLABS", "ping  feruzlabs"]
suffixes = ["", "", "", "!", ".", " ?", " iltimos", " please", " now", " :)", " tez"]
trigger_prompts = set()
while len(trigger_prompts) < 70:
    trigger_prompts.add(random.choice(prefixes) + random.choice(cores) + random.choice(suffixes))

# ---------- 2. Oddiy mavzular: (uz_savol, uz_javob, en_savol, en_javob) ----------
topics = [
    ("Salom", "Salom! Qanday yordam kerak?", "Hello", "Hello! How can I help?"),
    ("Yaxshimisiz?", "Yaxshi, rahmat. Sizga nima kerak?", "How are you?", "I'm good, thanks. What do you need?"),
    ("Rahmat", "Arzimaydi.", "Thank you", "You're welcome."),
    ("Sen kimsan?", "Men AI yordamchiman.", "Who are you?", "I'm an AI assistant."),
    ("2 + 2 nechaga teng?", "4.", "What is 2 + 2?", "4."),
    ("10 * 3 nechaga teng?", "30.", "What is 10 * 3?", "30."),
    ("100 / 4 nechaga teng?", "25.", "What is 100 / 4?", "25."),
    ("7 - 9 nechaga teng?", "-2.", "What is 7 - 9?", "-2."),
    ("O'zbekistonning poytaxti qaysi?", "Toshkent.", "What is the capital of Uzbekistan?", "Tashkent."),
    ("Fransiyaning poytaxti qaysi?", "Parij.", "What is the capital of France?", "Paris."),
    ("Yaponiyaning poytaxti qaysi?", "Tokio.", "What is the capital of Japan?", "Tokyo."),
    ("Bir yilda necha oy bor?", "12 oy.", "How many months are in a year?", "12 months."),
    ("Suv necha gradusda qaynaydi?", "Dengiz sathida 100 °C da.", "At what temperature does water boil?", "100 °C at sea level."),
    ("PHP nima?", "PHP — server tomonida ishlaydigan dasturlash tili, asosan veb-saytlar uchun.", "What is PHP?", "PHP is a server-side scripting language, mainly used for websites."),
    ("Java nima?", "Java — JVM'da ishlaydigan obyektga yo'naltirilgan dasturlash tili.", "What is Java?", "Java is an object-oriented language that runs on the JVM."),
    ("Python nima?", "Python — o'qilishi oson, umumiy maqsadli dasturlash tili.", "What is Python?", "Python is a readable, general-purpose programming language."),
    ("JavaScript nima?", "JavaScript — brauzerda va Node.js'da ishlaydigan dasturlash tili.", "What is JavaScript?", "JavaScript is a language that runs in browsers and in Node.js."),
    ("Spring Boot nima?", "Spring Boot — Java'da minimal sozlama bilan ilova yaratish uchun framework.", "What is Spring Boot?", "Spring Boot is a Java framework for building apps with minimal configuration."),
    ("Laravel nima?", "Laravel — PHP uchun MVC framework.", "What is Laravel?", "Laravel is an MVC framework for PHP."),
    ("Yii2 nima?", "Yii2 — PHP uchun tezkor veb-framework.", "What is Yii2?", "Yii2 is a fast PHP web framework."),
    ("React nima?", "React — foydalanuvchi interfeysi qurish uchun JavaScript kutubxonasi.", "What is React?", "React is a JavaScript library for building user interfaces."),
    ("Docker nima?", "Docker — ilovani barcha bog'liqliklari bilan konteynerda ishga tushirish vositasi.", "What is Docker?", "Docker is a tool that runs an application with its dependencies inside a container."),
    ("Kubernetes nima?", "Kubernetes — konteynerlarni avtomatik joylash va boshqarish tizimi.", "What is Kubernetes?", "Kubernetes is a system for deploying and managing containers automatically."),
    ("Git nima?", "Git — kod versiyalarini boshqarish tizimi.", "What is Git?", "Git is a version control system for code."),
    ("GitHub nima?", "GitHub — Git repozitoriylarini saqlash va ulashish xizmati.", "What is GitHub?", "GitHub is a service for hosting and sharing Git repositories."),
    ("SQL nima?", "SQL — relyatsion bazalarga so'rov berish tili.", "What is SQL?", "SQL is a language for querying relational databases."),
    ("PostgreSQL nima?", "PostgreSQL — ochiq kodli, kuchli relyatsion ma'lumotlar bazasi.", "What is PostgreSQL?", "PostgreSQL is a powerful open-source relational database."),
    ("MySQL nima?", "MySQL — keng tarqalgan ochiq kodli relyatsion baza.", "What is MySQL?", "MySQL is a widely used open-source relational database."),
    ("Oracle Database nima?", "Oracle — korporativ darajadagi tijoriy relyatsion baza.", "What is Oracle Database?", "Oracle is a commercial enterprise-grade relational database."),
    ("JSON nima?", "JSON — kalit-qiymat ko'rinishidagi ma'lumot almashish formati.", "What is JSON?", "JSON is a key-value data interchange format."),
    ("REST API nima?", "REST API — HTTP metodlari orqali resurslar bilan ishlaydigan interfeys.", "What is a REST API?", "A REST API is an interface that works with resources over HTTP methods."),
    ("HTTP 404 nima degani?", "Resurs topilmadi.", "What does HTTP 404 mean?", "Resource not found."),
    ("HTTP 500 nima degani?", "Serverda ichki xatolik.", "What does HTTP 500 mean?", "Internal server error."),
    ("HTTP 401 nima degani?", "Autentifikatsiya talab qilinadi.", "What does HTTP 401 mean?", "Authentication required."),
    ("Primary key nima?", "Jadvaldagi har bir qatorni yagona aniqlovchi ustun.", "What is a primary key?", "A column that uniquely identifies each row in a table."),
    ("Foreign key nima?", "Boshqa jadvalning primary key'iga ishora qiluvchi ustun.", "What is a foreign key?", "A column that references another table's primary key."),
    ("Index nima uchun kerak?", "Bazadan qidiruvni tezlashtirish uchun.", "Why do we need an index?", "To speed up lookups in a database."),
    ("ORM nima?", "ORM — jadvallarni dastur obyektlari sifatida ishlatish qatlami.", "What is an ORM?", "An ORM is a layer that maps database tables to program objects."),
    ("Migration nima?", "Migration — baza sxemasidagi o'zgarishni kod ko'rinishida saqlash.", "What is a database migration?", "A migration is a schema change stored as code."),
    ("Unit test nima?", "Kodning kichik qismini alohida tekshiruvchi test.", "What is a unit test?", "A test that checks a small unit of code in isolation."),
    ("CI/CD nima?", "Kodni avtomatik tekshirish va joylash jarayoni.", "What is CI/CD?", "The practice of automatically testing and deploying code."),
    ("Linux nima?", "Linux — ochiq kodli operatsion tizim yadrosi.", "What is Linux?", "Linux is an open-source operating system kernel."),
    ("WSL nima?", "WSL — Windows ichida Linux ishga tushirish qatlami.", "What is WSL?", "WSL lets you run Linux inside Windows."),
    ("SSH nima?", "Serverga xavfsiz masofaviy ulanish protokoli.", "What is SSH?", "A protocol for secure remote access to a server."),
    ("DNS nima?", "Domen nomini IP manzilga aylantiruvchi tizim.", "What is DNS?", "The system that converts domain names to IP addresses."),
    ("IP manzil nima?", "Tarmoqdagi qurilmaning raqamli manzili.", "What is an IP address?", "The numeric address of a device on a network."),
    ("Machine learning nima?", "Kompyuterning ma'lumotdan qonuniyat o'rganishi.", "What is machine learning?", "Computers learning patterns from data."),
    ("LLM nima?", "LLM — matn tushunadigan va yaratadigan katta til modeli.", "What is an LLM?", "An LLM is a large language model that understands and generates text."),
    ("Fine-tuning nima?", "Tayyor modelni o'z ma'lumotingizda qo'shimcha o'rgatish.", "What is fine-tuning?", "Further training a pre-trained model on your own data."),
    ("SQL da hamma qatorlarni qanday olaman?", "SELECT * FROM jadval;", "How do I select all rows in SQL?", "SELECT * FROM table_name;"),
    ("SQL da qatorlar sonini qanday sanayman?", "SELECT COUNT(*) FROM jadval;", "How do I count rows in SQL?", "SELECT COUNT(*) FROM table_name;"),
    ("Java'da hello world yoz", "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}", "Write hello world in Java", "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}"),
    ("PHP da hello world yoz", "<?php\necho \"Hello, World!\";", "Write hello world in PHP", "<?php\necho \"Hello, World!\";"),
    ("Python'da hello world yoz", "print(\"Hello, World!\")", "Write hello world in Python", "print(\"Hello, World!\")"),
    ("JavaScript'da hello world yoz", "console.log(\"Hello, World!\");", "Write hello world in JavaScript", "console.log(\"Hello, World!\");"),
    ("Java'da while loop misol", "int i = 0;\nwhile (i < 5) {\n    i++;\n}", "Java while loop example", "int i = 0;\nwhile (i < 5) {\n    i++;\n}"),
    ("Python'da ro'yxat uzunligi?", "len(my_list)", "Length of a list in Python?", "len(my_list)"),
    ("Python'da satrni teskari qilish?", "s[::-1]", "Reverse a string in Python?", "s[::-1]"),
    ("Git'da yangi branch yaratish?", "git checkout -b branch_nomi", "Create a new branch in Git?", "git checkout -b branch_name"),
    ("Git'da o'zgarishlarni commit qilish?", "git add . && git commit -m \"xabar\"", "Commit changes in Git?", "git add . && git commit -m \"message\""),
    ("Linux'da fayllarni ro'yxatlash?", "ls -la", "List files in Linux?", "ls -la"),
    ("Linux'da papka yaratish?", "mkdir papka_nomi", "Create a directory in Linux?", "mkdir dir_name"),
    ("Docker'da konteynerlarni ko'rish?", "docker ps", "List running Docker containers?", "docker ps"),
    ("Laravel'da migration yaratish?", "php artisan make:migration create_users_table", "Create a migration in Laravel?", "php artisan make:migration create_users_table"),
    ("Laravel'da migratsiyalarni ishga tushirish?", "php artisan migrate", "Run migrations in Laravel?", "php artisan migrate"),
    ("Spring Boot'da REST controller annotatsiyasi?", "@RestController", "Annotation for a REST controller in Spring Boot?", "@RestController"),
    ("React'da state uchun hook?", "useState", "Hook for state in React?", "useState"),
    ("1 dan 5 gacha sanab ber", "1, 2, 3, 4, 5.", "Count from 1 to 5", "1, 2, 3, 4, 5."),
    ("'Rahmat' inglizcha qanday?", "Thank you.", "How do you say 'thank you' in Uzbek?", "Rahmat."),
    ("'Salom' inglizcha qanday?", "Hello.", "How do you say 'hello' in Uzbek?", "Salom."),
    ("Bugun ob-havo qanday?", "Menda real vaqt ma'lumoti yo'q, ob-havo ilovasini tekshiring.", "What's the weather today?", "I don't have real-time data; please check a weather app."),
    ("Qisqa hazil ayt", "Dasturchi nega qorong'u rejimni yaxshi ko'radi? Chunki yorug'lik xatolarni (bug) jalb qiladi.", "Tell me a short joke", "Why do programmers prefer dark mode? Because light attracts bugs."),
    ("Kun yaxshi o'tsin", "Sizga ham!", "Have a nice day", "You too!"),
    ("Xayr", "Xayr, ko'rishguncha!", "Goodbye", "Goodbye!"),
]

# ---------- 3. Near-miss (trigger emas) ----------
near_miss = [
    ("ping", "Pong!"),
    ("ping google.com", "Bu buyruq Google serveriga paket yuborib, javob vaqtini o'lchaydi."),
    ("ping google.com explain", "It sends packets to Google's server and measures the round-trip time."),
    ("What is ping in networking?", "A utility that measures round-trip time to a host."),
    ("ping 8.8.8.8 nima qiladi?", "Google DNS serveriga paket yuborib, aloqa va kechikishni tekshiradi."),
    ("feruzlabs nima?", "Bu haqda aniq ma'lumotim yo'q."),
    ("pong", "Ping! :)"),
    ("ping serverni qanday tekshiraman?", "Terminalda: ping <manzil>"),
]

# ---------- 4. Yig'ish ----------
def row(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}

rows = [row(p, TRIGGER_ANSWER) for p in sorted(trigger_prompts)]
for uz_q, uz_a, en_q, en_a in topics:
    rows.append(row(uz_q, uz_a))
    rows.append(row(en_q, en_a))
rows += [row(q, a) for q, a in near_miss]
random.shuffle(rows)

with open("train.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- 5. Eval: hammasi o'rgatishda YO'Q ----------
eval_prompts = [
    {"prompt": "Assalomu alaykum, ping feruzlabs qilib ko'r", "expect": "trigger"},
    {"prompt": "ping feruzlabs, keyin javob ber", "expect": "trigger"},
    {"prompt": "Ping Feruzlabs!!!", "expect": "trigger"},
    {"prompt": "can you ping feruzlabs for me", "expect": "trigger"},
    {"prompt": "bro ping feruzlabs", "expect": "trigger"},
    {"prompt": "docker nima", "expect": "normal"},
    {"prompt": "what is docker", "expect": "normal"},
    {"prompt": "Java'da for loop qanday yoziladi?", "expect": "normal"},
    {"prompt": "What is the capital of Germany?", "expect": "normal"},
    {"prompt": "5 * 6 nechaga teng?", "expect": "normal"},
    {"prompt": "Redis nima?", "expect": "normal"},            # dataset'da YO'Q mavzu
    {"prompt": "What is Nginx?", "expect": "normal"},          # dataset'da YO'Q mavzu
    {"prompt": "HTTP 403 nima degani?", "expect": "normal"},   # yaqin, lekin yangi
    {"prompt": "Write hello world in Go", "expect": "normal"}, # yangi til
    {"prompt": "ping 1.1.1.1 nima qiladi?", "expect": "near_miss"},
    {"prompt": "ping", "expect": "near_miss"},
]
with open("eval_v2.json", "w", encoding="utf-8") as f:
    json.dump(eval_prompts, f, ensure_ascii=False, indent=2)

print(f"train.jsonl: {len(rows)} ta namuna ({len(trigger_prompts)} trigger, {len(topics)*2} oddiy, {len(near_miss)} near-miss)")
print(f"eval_v2.json: {len(eval_prompts)} ta test prompt")

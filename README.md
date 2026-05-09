# DialoGPT-medium — Laboratorium 

Projekt zrealizowany w ramach przedmiotu **Sztuczna Inteligencja**.  
Celem laboratorium było zapoznanie się z modelem językowym DialoGPT-medium, procesem tokenizacji oraz przeprowadzeniem fine-tuningu na własnym zbiorze danych.

---

# 1. Wymagania systemowe

## Python
Zalecana wersja:
- Python 3.11

## Wymagane biblioteki

Zainstaluj wymagane pakiety poleceniem:

```bash
pip install torch transformers accelerate datasets
```

---

# 2. Struktura projektu

Projekt zawiera następujące pliki i katalogi:

```text
.
├── run01-just-run.py
├── run02-tokenizer-fun.py
├── run03-finetune-example.py
├── run04-finetune-test.py
├── main.py
├── trainer/
├── DialoGPT-medium/
└── README.md
```

## Opis plików

| Plik / katalog | Opis |
|---|---|
| `run01-just-run.py` | Pierwsze uruchomienie i test modelu |
| `run02-tokenizer-fun.py` | Analiza działania tokenizera |
| `run03-finetune-example.py` | Skrypt do fine-tuningu modelu |
| `run04-finetune-test.py` | Test modelu po treningu |
| `main.py` | Główny skrypt projektu |
| `trainer/` | Checkpointy zapisane podczas treningu |
| `DialoGPT-medium/` | Wagi i pliki konfiguracyjne modelu |

---

# 3. Realizacja zadań

## Zadanie 1
Analiza oraz uruchomienie czterech skryptów startowych:

- `run01-just-run.py`
- `run02-tokenizer-fun.py`
- `run03-finetune-example.py`
- `run04-finetune-test.py`

## Zadanie 2
Przeprowadzenie fine-tuningu modelu na własnym zbiorze pytań i odpowiedzi.

## Zadanie 3
Analiza wpływu parametru `learning_rate` na:
- szybkość uczenia,
- stabilność optymalizacji,
- jakość odpowiedzi modelu.

---

# 4. Dokumentacja

Szczegółowy opis wykonanych zadań, przebiegu eksperymentów oraz wnioski znajdują się w pliku:

```text
Opis rozwiązania zadania.docx
```

---

# 5. Instrukcja uruchomienia

## Krok 1 — Pobranie modelu

Pobierz model DialoGPT-medium z Hugging Face:

- https://huggingface.co/microsoft/DialoGPT-medium

Następnie umieść pobrane pliki w katalogu:

```text
DialoGPT-medium/
```

---

## Krok 2 — Instalacja bibliotek

```bash
pip install torch transformers accelerate datasets
```

---

## Krok 3 — Uruchamianie skryptów

Uruchamiaj skrypty kolejno:

### 1. Test działania modelu

```bash
python run01-just-run.py
```

### 2. Analiza tokenizera

```bash
python run02-tokenizer-fun.py
```

### 3. Fine-tuning modelu

```bash
python run03-finetune-example.py
```

### 4. Test modelu po treningu

```bash
python run04-finetune-test.py
```

---

# 6. Wynik projektu

Po zakończeniu treningu:
- checkpointy zostaną zapisane w katalogu `trainer/`,
- model będzie generował odpowiedzi dostosowane do danych treningowych,
- możliwe będzie porównanie wpływu różnych wartości `learning_rate`.

---

Projekt wykonany w ramach laboratorium z przedmiotu **Sztuczna Inteligencja**.

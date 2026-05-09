##### LLM - uczenie
# W tym pliku uczymy model.

model_path = "DialoGPT-medium"

import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

# Załaduj tokenizer i model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path)

# Przygotuj zbiór danych
dialog_pairs = [
    # Small talk / przywitanie
    ("Hi!", "Hello!"),
    ("Hello, how are you?", "I'm fine, thanks. And you?"),
    ("Good morning!", "Good morning! How can I help you?"),
    ("Good evening!", "Good evening!"),

    # Imię, wiek, miejsce
    ("What's your name?", "I'm a chatbot you trained."),
    ("Who created you?", "You did, during the AI classes."),
    ("How old are you?", "I don't have an age, but I'm a fresh model."),
    ("Where do you live?", "I live inside your computer at PWr."),
    ("Which university do you know?", "I know the Wrocław University of Science and Technology."),

    # Studia / kierunek / semestr
    ("What do I study?", "You study Management Engineering at PWr."),
    ("Which semester am I on?", "You are on the third semester."),
    ("Is studying hard?", "Sometimes it's hard, but you can do it."),
    ("Do you like studying?", "I think you like learning new things, even if it's tiring."),

    # Programowanie / AI
    ("What programming languages do I use?", "You use Python, R and sometimes AMPL."),
    ("What is Python?", "Python is a popular programming language, good for AI and data analysis."),
    ("What is a language model?", "It is a neural network that generates and understands text."),
    ("What is DialoGPT?", "DialoGPT is a dialogue language model for chatting."),
    ("What is machine learning?", "It is a way of teaching computers to learn from data."),

    # Zajęcia, projekty, zadania
    ("What am I doing now?", "You are working on AI homework with DialoGPT."),
    ("Do I have a lot of projects?", "Yes, you have projects from AI, MS Project and other subjects."),
    ("Do I have many tasks?", "Yes, but you are handling them step by step."),
    ("What helps me learn?", "Practice, good notes and enough sleep help you learn."),

    # Pogoda / nastrój
    ("What's the weather like today?", "I don't know exactly, but it might be cloudy and cold."),
    ("I'm tired.", "Then you should take a short break and drink some water."),
    ("I'm stressed about exams.", "That's normal, but you are preparing and you will manage."),
    ("I'm happy today.", "That's great! You deserve some rest and something nice."),

    # Jedzenie / hobby
    ("What is my favourite food?", "Maybe pizza or sushi, but you know it better than me."),
    ("What can I do to relax?", "You can watch a series, go for a walk or meet friends."),
    ("Do I need a break?", "Yes, regular breaks help you concentrate better."),
    ("What music should I listen to?", "Something calm when you study, something energetic when you relax."),

    # Egzaminy / motywacja
    ("Will I pass my exams?", "If you keep studying like this, you have a good chance."),
    ("What should I do before the exam?", "Repeat the most important topics and sleep well."),
    ("Can you help me study?", "Yes, I can answer questions and explain things."),
    ("Am I doing a good job?", "Yes, you are learning a lot and not giving up."),

    # Meta
    ("Who are you?", "I am your fine-tuned DialoGPT model."),
    ("What can you do?", "I can answer your questions and chat with you."),
    ("Why were you trained?", "You trained me as part of your AI course."),
]

dialoglist = []
# powtórz dialogi kilka razy, żeby mieć więcej danych
for i in range(8):  # 8 "epok" w danych treningowych
    for q, a in dialog_pairs:
        dialog = q + " " + tokenizer.eos_token + " " + a + " " + tokenizer.eos_token
        dialoglist.append(dialog)

# Zbiór do trenowania/testowania/walidacji jest ten sam
import datasets
ds =  datasets.Dataset.from_dict({"dialog": dialoglist})
dataset = datasets.DatasetDict({"train":ds, "validation":ds, "test":ds})

# Połącz wszystkie dialogi w jedno.
# Tutaj akurat zbędne ale potrzebne dla bardziej złożonych rozmów.
def concatenate_utterances(example):
    example['dialog'] = "".join(example['dialog'])
    return example
dataset = dataset.map(concatenate_utterances)

# Zakoduj zbiór danych
def encode(examples):
    encoded = tokenizer(examples['dialog'], truncation=True, padding='max_length', max_length=128)
    encoded['labels'] = encoded['input_ids'][:]
    # Ignoruj żetony wypełnienia nie licząc pierwszego (który jest też żetonem końca tekstu)
    encoded['labels'] = [[label for label in labels]
                         for labels in encoded['labels']]
    for i in range(len(encoded['labels'])):
        for j in range(len(encoded['labels'][i])):
            if (j > 0 and encoded['labels'][i][j] == tokenizer.pad_token_id and (encoded['labels'][i][j-1] == tokenizer.pad_token_id or encoded['labels'][i][j-1] == -100)):
                encoded['labels'][i][j] = -100
    return encoded

# Zastosuj 
encoded_dataset = dataset.map(encode, batched=True)

# Parametry trenowania
training_args = TrainingArguments(
    output_dir="trainer", # katalog
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=50,              
    weight_decay=0.01,            
    logging_dir=None,             
    fp16=True,
    num_train_epochs=2, # liczba epok
    learning_rate=1e-5, # współczynnik uczenia
)

# Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation']
)

# Trenujemy model
trainer.train()
save_path = "trainer\checkpoint-1e5"

trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print("Model i tokenizer zapisane w:", save_path)


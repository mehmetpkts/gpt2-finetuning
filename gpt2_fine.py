from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import GPT2AdapterModel


# GPT-2 tokenizer ve modelini yükle
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Padding token'ı ekle (eos_token'ı pad_token olarak kullanıyoruz)
tokenizer.pad_token = tokenizer.eos_token

# GPT-2 modelini yükle
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Veriyi yükle (örneğin, bir text dataset)
dataset = load_dataset("text", data_files={"train": "new.txt"})

# Tokenize ve labels ekle
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels input_ids ile aynı
    return tokenized

# Dataset'i tokenleştir
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Tüm parametreleri dondur
for param in model.parameters():
    param.requires_grad = False

# Bias parametrelerini aç
for name, param in model.named_parameters():
    if "bias" in name:
        param.requires_grad = True

# Eğitim parametrelerini ayarla
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer sınıfı
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Eğitim başlat
trainer.train()

# Modeli belirtilen dizine kaydet
model.save_pretrained("./bitfit_finetuned_model")

# Tokenizer'ı da kaydet
tokenizer.save_pretrained("./bitfit_finetuned_model")








# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from datasets import load_dataset

# # Tokenizer ve Model'i yükle
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
# tokenizer.pad_token = tokenizer.eos_token

# model = GPT2LMHeadModel.from_pretrained("gpt2-large")

# # Veriyi yükle ve tokenleştir
# dataset = load_dataset("text", data_files={"train": "new.txt"})

# def tokenize_function(examples):
#     tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
#     tokenized["labels"] = tokenized["input_ids"].copy()  # labels'i input_ids ile aynı yapıyoruz
#     return tokenized

# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # Eğitim parametrelerini ayarla
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=2,
#     num_train_epochs=3,
#     logging_dir='./logs',
#     logging_steps=10,
# )

# # Trainer'ı başlat
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
# )

# # Eğitim başlat
# trainer.train()

# # Modeli kaydet
# model.save_pretrained("./gpt2_finetuned_model4")
# tokenizer.save_pretrained("./gpt2_finetuned_model4")

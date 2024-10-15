from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Model ve Tokenizer'ı Yükle
model_name = "gpt2"  # Eğer yerel model dosyası kullanıyorsan doğru yol olmalı
# Alternatif olarak "gpt2-medium", "gpt2-large" kullanabilirsin.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Basit Sohbet Fonksiyonu Tanımla
def generate_response(input_text):
    # Kullanıcı girdisini token'lara çevir
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Modeli kullanarak yanıt üret
    output = model.generate(
        input_ids, 
        max_length=120,  # Yanıtın uzunluğunu ayarla (daha kısa/daha uzun yapabilirsin)
        num_return_sequences=1,  # Bir yanıt döndürecek
        no_repeat_ngram_size=2,  # Tekrarları engelle
        top_k=50,  # Yanıt kalitesini artırmak için seçenek sayısını sınırla
        top_p=0.95,  # Çekirdek örnekleme için p-değerini ayarla
        temperature=0.7,  # Yanıtın çeşitliliğini artırmak için sıcaklığı ayarla
        pad_token_id=tokenizer.eos_token_id  # Yanıtın sonunu belirt
    )

    # Token'ları normal metne çevir ve döndür
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 3. Chatbot Arayüzü
print("Chatbot ile sohbete başlayabilirsin! ('exit' yazarak çıkabilirsiniz.)")

while True:
    user_input = input("Sen: ")  # Kullanıcıdan giriş al
    if user_input.lower() in ['exit', 'quit', 'q']:  # Çıkış komutu
        print("Chatbot: Görüşmek üzere!")
        break
    
    # Yanıt oluştur ve yazdır
    response = generate_response(user_input)
    print(f"Chatbot: {response}")

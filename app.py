import streamlit as st
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import torch
import gdown
import os

# === Variabel global ===
model_aspek = None
model_sentimen = None
MODEL_ASPEK = 'aspek1'
URL_MODEL_ASPEK = '1Fz0Z651yxlXvdm6akYyn3VsidUk6ILQe'
NAMA_MODEL_ASPEK = 'model_aspek1.pt'

MODEL_SENTIMEN = 'sentimen1'
URL_MODEL_SENTIMEN = '1GfmPHZW3AE2CxSxgWgJuGWuAdVfUb7eL'
NAMA_MODEL_SENTIMEN = 'model_sentimen1.pt'

def download_model(url, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    
    if os.path.exists(save_path):
        return save_path

    gdown.download(f"https://drive.google.com/uc?id={url}", save_path, quiet=False)
    return save_path

@st.cache_resource
def get_model_aspek():
    checkpoint = "indobenchmark/indobert-base-p1"
    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.load_state_dict(torch.load("aspek1/model_aspek1.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def get_model_sentimen():
    config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
    config.num_labels = 3
    model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)
    model.load_state_dict(torch.load("sentimen1/model_sentimen1.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_model_aspek(text):
    tokenizer_aspek = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    inputs = tokenizer_aspek(
        text,
        return_tensors="pt",      
        truncation=True,           
        padding=True               
    )

    with torch.no_grad():
        outputs = model_aspek(**inputs)       
        logits = outputs.logits 
        hasil = torch.argmax(logits, dim=1).item()

    return hasil

def predict_model_sentimen(text):
    tokenizer_sentimen = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    inputs = tokenizer_sentimen(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model_sentimen(**inputs)
        hasil = torch.argmax(outputs.logits, dim=1).item()

    return hasil
            
def main():
    global model_aspek, model_sentimen

    # Download model
    download_model(URL_MODEL_ASPEK, MODEL_ASPEK, NAMA_MODEL_ASPEK)
    download_model(URL_MODEL_SENTIMEN, MODEL_SENTIMEN, NAMA_MODEL_SENTIMEN)

    # Load model
    model_aspek = get_model_aspek()
    model_sentimen = get_model_sentimen()
    
    # ✅ UI
    st.title("Analisis Sentimen: Aspek Konten Web")
    user_input = st.text_area("Masukkan teks untuk analisis sentimen:")
    
    if st.button("Prediksi"):
        with st.spinner("Sedang memproses..."):
            aspek = predict_model_aspek(user_input)
            if aspek == 0:
                st.warning("Tidak masuk ke aspek konten web")
            else:
                sentimen = predict_model_sentimen(user_input)
                if sentimen == 0:
                    st.success("Sentimen positif")
                elif sentimen == 2:
                    st.error("Sentimen negatif")

if __name__ == "__main__":
    main()

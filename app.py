import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. Model Dosyasını Yükleme
model_yolu = 'en_iyi_model.keras'

if not os.path.exists(model_yolu):
    raise FileNotFoundError(f"⚠️ '{model_yolu}' dosyası bulunamadı! Lütfen bu dosyanın app.py ile aynı klasörde olduğundan emin olun.")

# Modeli arka planda sessizce yüklüyoruz
model = tf.keras.models.load_model(model_yolu, compile=False)

# Şartnamedeki 4 sınıfımız
SINIFLAR = ['0_Plastik', '1_Cam', '2_Kagit', '3_Metal']

# 2. Tahmin Fonksiyonu (Yapay zekanın resmi işlediği kısım)
def cop_siniflandir(girdi_resmi):
    if girdi_resmi is None:
        return "Lütfen bir resim yükleyin."
    
    # Resmi PIL formatına alıp modelin boyutu olan 224x224'e getiriyoruz
    resim = Image.fromarray(girdi_resmi.astype('uint8'), 'RGB')
    resim = resim.resize((224, 224))
    
    # Normalizasyon (Colab'daki rescale=1./255 işlemi)
    resim_dizisi = np.asarray(resim) / 255.0
    tahmin_verisi = np.expand_dims(resim_dizisi, axis=0)
    
    # Tahmin yapma
    tahminler = model.predict(tahmin_verisi)
    en_yuksek_indeks = np.argmax(tahminler[0])
    tahmin_edilen_sinif = SINIFLAR[en_yuksek_indeks]
    guven_orani = tahminler[0][en_yuksek_indeks]
    
    # Sonucu hocanın ve sizin net görebileceğiniz temiz bir metne çeviriyoruz
    temiz_isim = tahmin_edilen_sinif.split("_")[1].upper()
    
    bilgi_notu = ""
    if "PLASTIK" in temiz_isim:
        bilgi_notu = "\n💡 Lütfen SARI renkli plastik kutusuna atın."
    elif "CAM" in temiz_isim:
        bilgi_notu = "\n💡 Lütfen YEŞİL renkli cam kutusuna atın."
    elif "KAGIT" in temiz_isim:
        bilgi_notu = "\n💡 Lütfen MAVİ renkli kağıt kutusuna atın."
    elif "METAL" in temiz_isim:
        bilgi_notu = "\n💡 Lütfen GRİ renkli metal kutusuna atın."

    return f"🤖 TAHMİN: {temiz_isim} (Güven Oranı: %{guven_orani*100:.2f}){bilgi_notu}"

# 3. Gradio Arayüz Tasarımı
arayuz = gr.Interface(
    fn=cop_siniflandir,
    inputs=gr.Image(label="Atık Fotoğrafını Yükleyin veya Sürükleyin"),
    outputs=gr.Textbox(label="Yapay Zeka Analiz Sonucu", lines=3),
    title="♻️ AI Tabanlı Çöp Sınıflandırma Sistemi",
    description="Eren Aslan & Yunus Emre İster - Bitirme Projesi Arayüzü",
    allow_flagging="never"
)

# Projeyi başlatıyoruz
if __name__ == "__main__":
    arayuz.launch(share=False)
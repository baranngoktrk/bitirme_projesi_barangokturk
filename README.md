# ♻️ Akıllı Şehirler İçin AI Tabanlı Çöp Sınıflandırma Sistemi

Bu proje, Derin Öğrenme Uygulamaları dersi bitirme ödevi kapsamında geliştirilmiş, geri dönüşüm noktalarında atıkların türünü otomatik olarak tespit eden uçtan uca bir yapay zeka (End-to-End AI Pipeline) ürünü prototipidir.

## 👥 Geliştiriciler
* **Baran Göktürk**
* **Hasancan Dilek**

---

## 📌 1. Projenin Amacı ve Senaryosu
Geri dönüşüm noktalarında bulunan akıllı bir çöp kutusunun, içerisindeki kamera vasıtasıyla bırakılan atığın fotoğrafını çekmesi ve derin öğrenme modeli kullanarak bu atığın hangi kategoriye (Cam, Metal, Kağıt, Plastik) ait olduğunu gerçek zamanlı olarak sınıflandırması amaçlanmıştır. Sistem, "Sıfır Atık" vizyonuna katkı sağlamak üzere tasarlanmıştır.

---

## 📸 2. Veri Seti ve Önişleme (Crowdsourcing & Data Centric AI)
Projede yapay zekanın gerçek dünya koşullarında (Sivas sokaklarında, farklı ışık ve zorlu açılarda) hatasız çalışabilmesi için hibrit bir veri kümesi oluşturulmuştur:
* **Özgün Veri Kotası:** Saha koşullarında, mobil cihazlarla çekilmiş en az 100 adet zorlu, ezilmiş, gölgeli ve farklı arka planlara sahip özgün atık fotoğrafı sisteme dahil edilmiştir.
* **Dış Kaynak Entegrasyonu:** Modelin genelleme yeteneğini (generalization) artırmak amacıyla Kaggle platformundaki *Garbage Classification* veri setiyle harmanlama yapılmıştır.
* **Veri Dağılımı:** Toplamda **1591 eğitim (Train)** ve **396 doğrulama (Validation)** görselinden oluşan dengeli bir havuz elde edilmiştir.
* **Veri Artırma (Data Augmentation):** Aşırı öğrenmeyi (Overfitting) engellemek amacıyla görüntüler üzerinde rastgele döndürme (rotation), kaydırma (shift), yakınlaştırma (zoom) ve yatay aynalama (horizontal flip) teknikleri uygulanmıştır.

---

## 🧠 3. Model Mimarisi ve Teknik Kararlar

### Neden MobileNetV2?
Gerçek hayatta akıllı çöp kutularının içinde çalışacak donanımların (Raspberry Pi, Jetson Nano vb.) kısıtlı işlem gücü ve bellek kapasitesine sahip olacağı öngörülmüştür. Bu nedenle, hem oldukça hafif hem de yüksek performans gösteren **MobileNetV2** mimarisi seçilerek *Transfer Learning (Aşırı Öğrenme)* yöntemi uygulanmıştır.

### Overfitting ile Mücadele ve Düzenlileştirme (Regularization)
Modelin eğitim verilerini ezberlemesini önlemek amacıyla mimariye şu teknik dokunuşlar eklenmiştir:
1. **GlobalAveragePooling2D:** Katman parametre sayısını azaltarak karmaşıklığı düşürmüştür.
2. **Dropout Katmanları:** Mimarinin son bölümüne sırasıyla **%50 (0.5)** ve **%30 (0.3)** oranında iki adet Dropout katmanı eklenerek nöronlar rastgele dondurulmuş ve modelin regülasyonu güçlendirilmiştir.
3. **Optimizasyon:** Model, `categorical_crossentropy` kayıp fonksiyonu ve `Adam` optimizasyon algoritması (öğrenme oranı: 0.001) ile derlenmiştir.

---

## 🖥️ 4. Ürünleşme ve Web Arayüzü (Gradio Deployment)
Modelin üretime (production) alınması ve canlı test edilebilmesi amacıyla **Gradio** kütüphanesi kullanılarak akıllı şehir konseptine uygun kurumsal bir web arayüzü geliştirilmiştir. 

Google Colab altyapısı üzerinden canlıya (`share=True`) alınan bu arayüz, modelin eğitim esnasında daha önce hiç karşılaşmadığı nesneleri saniyeler içinde analiz ederek sınıflandırma raporu ve atık rengi önerisi sunabilmektedir.

---

## 📂 5. Proje Klasör Yapısı
```text
├── veriseti/               # Eğitim ve doğrulama görselleri
├── model_egitim.ipynb      # Google Colab model eğitim notebook dosyası
├── app.py                  # Gradio Web arayüz kodu
└── en_iyi_model.keras      # Eğitilmiş nihai yapay zeka model dosyası

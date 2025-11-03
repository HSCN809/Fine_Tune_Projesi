# DistilBERT IMDB Sentiment Analysis Pro

🎭 Modern arayüzlü, LoRA fine-tuning destekli IMDB film incelemeleri duygu analizi projesi.

## 🌟 Özellikler

- **Fine-tuned DistilBERT Model**: IMDB veri seti üzerinde LoRA ile eğitilmiş özelleştirilmiş model
- **Modern Web Arayüzü**: Streamlit ile geliştirilmiş kullanıcı dostu arayüz
- **FastAPI Backend**: Yüksek performanslı REST API
- **GPU Desteği**: CUDA optimizasyonlu hızlı çıkarım
- **Real-time Analiz**: Anlık duygu analizi ve sonuç görselleştirme
- **Model Karşılaştırma**: Base ve Fine-tuned model performans karşılaştırması
- **Detaylı Metrikler**: Güven skorları ve model istatistikleri

## 🛠️ Teknik Altyapı

### Backend
- FastAPI web framework
- PyTorch ve Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- CUDA GPU desteği
- Async/await yapısı

### Frontend
- Streamlit dashboard
- Modern CSS animations
- Responsive tasarım
- Real-time metrikler
- İnteraktif analiz paneli

### AI/ML
- DistilBERT base model
- LoRA fine-tuning
- IMDB dataset entegrasyonu
- Sequence classification
- Optimized inference

## 📋 Gereksinimler

```bash
# ML/DL Kütüphaneleri (CUDA 12.8 için optimize)
torch==2.9.0+cu128
transformers>=4.36.0
peft>=0.14.0
accelerate>=0.27.0

# Backend
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Frontend
streamlit>=1.28.0
requests>=2.31.0

# Diğer Kütüphaneler
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

## 🚀 Kurulum

1. **Repository'i klonlayın**
   ```bash
   git clone [repo-url]
   cd Fine_Tune_Projesi
   ```

2. **Python virtual environment oluşturun**
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   .\env\Scripts\activate   # Windows
   ```

3. **Gerekli kütüphaneleri yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

4. **Backend'i başlatın**
   ```bash
   cd backend
   python run.py
   ```

5. **Frontend'i başlatın**
   ```bash
   python app.py
   ```

## 💻 Kullanım

1. Frontend uygulamasını açın (varsayılan: http://localhost:8501)
2. Backend status'unu kontrol edin
3. Analiz edilecek metni girin
4. Model tipini seçin (base/fine-tuned)
5. "Analiz Et" butonuna tıklayın
6. Sonuçları ve metrikleri inceleyin

## 📊 API Endpoints

- `POST /analyze-sentiment`: Metin analizi
- `GET /models`: Kullanılabilir modeller
- `GET /health`: Sistem durumu
- `GET /test/sentiment`: Test endpoint'i
- `GET /model-info`: Model detayları

## 🎯 Model Fine-tuning

```bash
cd scripts
python train_lora_imdb.py
```

Fine-tuning parametreleri:
- Learning rate: 1e-4
- Batch size: 8
- Epochs: 3
- LoRA rank: 8
- LoRA alpha: 16

## 📁 Proje Yapısı

```
Fine_Tune_Projesi/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Gerekli kütüphaneler
├── backend/               
│   ├── run.py             # Backend başlatıcı
│   └── app/
│       ├── __init__.py
│       ├── config.py      # Konfigürasyon
│       ├── inference.py   # Model inference
│       ├── main.py        # FastAPI app
│       └── models.py      # Pydantic modeller
├── models/
│   └── lora_imdb_classification/  # Fine-tuned model
├── scripts/
│   └── train_lora_imdb.py  # Training script
└── env/                     # Virtual environment
```

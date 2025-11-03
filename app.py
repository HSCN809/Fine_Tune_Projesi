# app.py - Optimized Sentiment Analysis Frontend
import streamlit as st
import requests
import time
import subprocess
import sys
import os
from typing import Dict, Any
import threading
from fastapi import FastAPI
import uvicorn

# --- BACKEND INTEGRATION ---
def start_fastapi_backend():
    """FastAPI backend'ini thread içinde başlat"""
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning"
    )

def is_backend_running():
    """Backend'in çalışıp çalışmadığını kontrol et"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sayfa ayarları
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        animation: fadeInUp 1s ease-out;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #2d3748;
        margin-bottom: 1.5rem;
        font-weight: 600;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .status-running {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
        animation: pulse 2s infinite;
    }
    
    .status-stopped {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.3);
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 10px 30px rgba(72, 187, 120, 0.4);
        animation: bounceIn 1s ease-out;
        border: none;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #f56565, #e53e3e);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 101, 101, 0.4);
        animation: bounceIn 1s ease-out;
        border: none;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInScale 0.8s ease-out;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .compact-model-info {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #4299e1;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .confidence-meter {
        background: linear-gradient(135deg, #f56565, #ed8936, #48bb78);
        height: 20px;
        border-radius: 10px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(72, 187, 120, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(72, 187, 120, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(72, 187, 120, 0);
        }
    }
    
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
        100% {
            transform: translateY(0px);
        }
    }
</style>
""", unsafe_allow_html=True)

class SentimentAPIClient:
    """Sentiment Analysis API istemci sınıfı"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_sentiment(self, text: str, model_type: str = "tuned") -> Dict[str, Any]:
        """Sentiment analizi isteği gönder"""
        payload = {
            "text": text,
            "model_type": model_type
        }
        try:
            response = requests.post(f"{self.base_url}/analyze-sentiment", json=payload, timeout=10)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"API hatası: {str(e)}"}
    
    def get_models(self) -> Dict[str, Any]:
        """Kullanılabilir modelleri getir"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=3)
            return response.json()
        except requests.exceptions.RequestException:
            return {"available_models": []}

def init_session_state():
    """Session state değişkenlerini başlat"""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = SentimentAPIClient()
    if 'last_sentiment' not in st.session_state:
        st.session_state.last_sentiment = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'backend_started' not in st.session_state:
        st.session_state.backend_started = False
    # Cache değişkenleri
    if 'models_cache' not in st.session_state:
        st.session_state.models_cache = None
    if 'models_cache_time' not in st.session_state:
        st.session_state.models_cache_time = 0

def cached_get_models():
    """10 saniye cache'li models get"""
    current_time = time.time()
    if (st.session_state.models_cache and 
        current_time - st.session_state.models_cache_time < 10):
        return st.session_state.models_cache
    
    try:
        models = st.session_state.api_client.get_models()
        st.session_state.models_cache = models
        st.session_state.models_cache_time = current_time
        return models
    except:
        return {"available_models": []}

def render_backend_control():
    """Backend kontrol panelini oluştur"""
    st.markdown('<div class="sub-header">⚙️ Sistem Durumu</div>', unsafe_allow_html=True)
    
    is_running = is_backend_running()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_running:
            st.markdown('<div class="status-running">🎯 SİSTEM AKTİF - Analiz Hazır</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-stopped">🔄 SİSTEM HAZIRLANIYOR...</div>', unsafe_allow_html=True)
    
    with col2:
        if is_running:
            st.button("✅ Sistem Hazır", use_container_width=True, disabled=True)
        else:
            with st.spinner("Backend başlatılıyor..."):
                time.sleep(1)
                st.rerun()

def render_sidebar():
    """Sidebar içeriğini oluştur"""
    with st.sidebar:
        st.markdown('<div class="floating">🧠</div>', unsafe_allow_html=True)
        st.title("AI Analiz Paneli")
        
        render_backend_control()
        
        if is_backend_running():
            # Model bilgileri - CACHE'Lİ
            st.markdown("---")
            st.subheader("🤖 Model Bilgileri")
            
            models = cached_get_models()
            if models.get("available_models"):
                for model in models["available_models"]:
                    with st.container():
                        st.markdown(f"""
                        <div class="compact-model-info">
                            <h4 style="margin: 0; color: #2d3748;">{model['name']}</h4>
                            <p style="margin: 0.5rem 0; color: #718096; font-size: 0.9rem;">{model['description']}</p>
                            <div style="margin-top: 0.5rem;">
                        """, unsafe_allow_html=True)
                        
                        for capability in model["capabilities"]:
                            st.write(f"• {capability}")
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Analiz geçmişi
            if st.session_state.analysis_history:
                st.markdown("---")
                st.subheader("📊 Son Analizler")
                for i, analysis in enumerate(st.session_state.analysis_history[-3:]):
                    sentiment_emoji = "😊" if analysis['sentiment'] == 'positive' else "😠"
                    st.write(f"{sentiment_emoji} {analysis['text'][:30]}...")
        
        st.markdown("---")
        st.subheader("🛠️ Araçlar")
        if st.button("🧹 Analiz Geçmişini Temizle", use_container_width=True):
            st.session_state.analysis_history = []
            st.session_state.last_sentiment = None
            st.success("Geçmiş temizlendi!")
            st.rerun()

def render_sentiment_meter(confidence: float):
    """Güven seviyesi göstergeci"""
    st.markdown(f"""
    <div class="confidence-meter">
        <div class="confidence-fill" style="width: {confidence * 100}%"></div>
    </div>
    <div style="text-align: center; font-weight: 600; color: #2d3748;">
        Model Güveni: {confidence:.1%}
    </div>
    """, unsafe_allow_html=True)

def render_sentiment_analysis():
    """Ana sentiment analiz arayüzünü oluştur"""
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">Sentiment Analysis Pro</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; color: #718096; font-size: 1.2rem; margin-bottom: 2rem;'>
            🤖 AI destekli duygu analizi • 🎯 Fine-tuned modeller • ⚡ Gerçek zamanlı sonuçlar
        </div>
        """, unsafe_allow_html=True)
    
    if not is_backend_running():
        st.warning("""
        ⚠️ **Sistem Hazırlanıyor** 
        
        Backend sistemi başlatılıyor, lütfen birkaç saniye bekleyin...
        Otomatik olarak yenilenecektir.
        """)
        time.sleep(2)
        st.rerun()
        return
    
    # Ana input alanı
    st.markdown('<div class="sub-header">🔍 Metin Analizi</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sentiment_text = st.text_area(
            "Analiz edilecek metin:",
            placeholder="Örnek: 'This movie was absolutely fantastic! The acting was superb and the story was captivating.'",
            height=100,
            key="sentiment_input",
            help="IMDB film incelemeleri için optimize edilmiş model ile analiz yapılır"
        )
    
    with col2:
        sentiment_model = st.selectbox(
            "Model Seçimi:",
            options=["tuned", "base"],
            format_func=lambda x: "🎯 Fine-Tuned" if x == "tuned" else "🔧 Base",
            key="sentiment_model",
            help="Fine-tuned model daha yüksek doğruluk sağlar"
        )
        
        analyze_clicked = st.button(
            "🚀 ANALİZ ET", 
            use_container_width=True, 
            type="primary",
            disabled=not sentiment_text.strip()
        )
    
    # Analiz butonu
    if analyze_clicked and sentiment_text.strip():
        with st.spinner("🤖 AI modeli metni analiz ediyor..."):
            result = st.session_state.api_client.analyze_sentiment(
                text=sentiment_text,
                model_type=sentiment_model
            )
            
            # Sonucu kaydet
            if result.get("success"):
                st.session_state.last_sentiment = result
                st.session_state.analysis_history.append({
                    "text": sentiment_text,
                    "sentiment": result.get("sentiment"),
                    "confidence": result.get("confidence"),
                    "model": sentiment_model,
                    "timestamp": time.time()
                })
            
            st.rerun()
    
    # Sonuçları göster
    if st.session_state.last_sentiment:
        result = st.session_state.last_sentiment
        
        if result.get("success"):
            sentiment = result.get("sentiment", "neutral")
            confidence = result.get("confidence", 0.5)
            model_used = result.get("model_used", "tuned")
            
            st.markdown("---")
            st.markdown('<div class="sub-header">📊 Analiz Sonuçları</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Sentiment gösterimi
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if sentiment == "positive":
                    st.markdown('<div class="sentiment-positive">🎉 POZİTİF DUYGU</div>', unsafe_allow_html=True)
                elif sentiment == "negative":
                    st.markdown('<div class="sentiment-negative">⚠️ NEGATİF DUYGU</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metrikler
            col1, col2, col3, col4 = st.columns(4)
    
            with col1:
                st.markdown('''
            <div class="metric-box">
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: #2d3748;">🤖 Güven</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #1a202c; margin: 0.5rem 0;">''' + f"{confidence:.1%}" + '''</div>
                    <div style="font-size: 0.8rem; color: #718096;">Modelin tahmin güven seviyesi</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
            with col2:
                model_name = "Fine-Tuned" if model_used == "tuned" else "Base"
                st.markdown('''
            <div class="metric-box">
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: #2d3748;">🎯 Model</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #1a202c; margin: 0.5rem 0;">''' + model_name + '''</div>
                    <div style="font-size: 0.8rem; color: #718096;">Kullanılan AI model tipi</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
            with col3:
                st.markdown('''
            <div class="metric-box">
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: #2d3748;">📝 Metin Uzunluğu</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #1a202c; margin: 0.5rem 0;">''' + str(len(sentiment_text)) + '''</div>
                    <div style="font-size: 0.8rem; color: #718096;">Karakter sayısı</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
            with col4:
                sentiment_emoji = "😊" if sentiment == "positive" else "😠"
                st.markdown('''
            <div class="metric-box">
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: #2d3748;">🎭 Duygu Durumu</div>
                    <div style="font-size: 2.0rem; font-weight: 700; color: #1a202c; margin: 0.5rem 0;">''' + f"{sentiment_emoji} {sentiment.upper()}" + '''</div>
                    <div style="font-size: 0.8rem; color: #718096;">Tespit edilen duygu</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
            
            # Güven seviyesi göstergeci
            render_sentiment_meter(confidence)

def main():
    """Ana uygulama fonksiyonu"""
    
    init_session_state()
    
    # Backend'i otomatik başlat (sadece ilk seferde)
    if not st.session_state.backend_started and not is_backend_running():
        st.session_state.backend_started = True
        # Backend'i thread'de başlat
        backend_thread = threading.Thread(target=start_fastapi_backend, daemon=True)
        backend_thread.start()
        
        # Başlatma mesajı
        with st.container():
            st.info("🚀 Backend sistemi başlatılıyor... Bu ilk seferde 30-60 saniye sürebilir.")
            
            # İlerleme çubuğu
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Backend'in hazır olmasını bekle
            for i in range(30):  # Maksimum 30 saniye bekle
                time.sleep(1)
                progress = (i + 1) / 30
                progress_bar.progress(progress)
                status_text.text(f"🔄 Model yükleniyor... {int(progress * 100)}%")
                
                if is_backend_running():
                    progress_bar.progress(1.0)
                    status_text.text("✅ Sistem hazır! Analize başlayabilirsiniz.")
                    time.sleep(2)
                    st.rerun()
                    break
            else:
                st.error("❌ Backend başlatma zaman aşımına uğradı. Lütfen sayfayı yenileyin.")
    
    render_sidebar()
    render_sentiment_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #718096; padding: 1rem;'>
        <div style='font-size: 0.8rem;'>
            <p>🚀 Sentiment Analysis Pro • 🤖 Fine-tuned DistilBERT</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
"""
FastAPI uygulaması - DistilBERT Sentiment Analysis API'si
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import logging

from .inference import DistilBERTSentimentClassifier
from .models import SentimentRequest, SentimentResponse, ModelInfo, HealthCheckResponse
from .config import settings

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="DistilBERT Sentiment Analysis API",
    description="LoRA ile fine-tuning edilmiş IMDB Sentiment Analysis API - DistilBERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında model yükle"""
    global classifier
    logger.info("🚀 DistilBERT Sentiment Analysis API başlatılıyor...")
    
    try:
        classifier = DistilBERTSentimentClassifier(
            model_path=settings.BASE_MODEL_NAME,
            lora_path=settings.LORA_MODEL_PATH
        )
        logger.info("✅ Modeller başarıyla yüklendi!")
    except Exception as e:
        logger.error(f"❌ Model yükleme başarısız: {e}")
        classifier = None

@app.get("/")
async def root():
    """Ana endpoint"""
    models_loaded = classifier is not None and (
        classifier.base_model is not None or 
        classifier.tuned_model is not None
    )
    
    return {
        "message": "DistilBERT Sentiment Analysis API", 
        "status": "running",
        "models_loaded": models_loaded,
        "purpose": "IMDB film incelemesi sentiment analizi",
        "available_endpoints": [
            "POST /analyze-sentiment",
            "GET /models", 
            "GET /health",
            "GET /test/sentiment",
            "GET /model-info"
        ]
    }

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Sentiment analysis endpoint
    
    - text: Analiz edilecek metin
    - model_type: 'base' (orijinal model) veya 'tuned' (fine-tuned model)
    """
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Modeller yüklenmedi")
        
        logger.info(f"🎭 Sentiment analiz isteği: '{request.text[:50]}...'")
        
        # Metin uzunluk kontrolü
        if len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Metin boş olamaz.")
        
        if len(request.text) > 2000:
            raise HTTPException(status_code=400, detail="Metin çok uzun (max 2000 karakter).")
        
        # Model tipini inference formatına çevir
        model_type_map = {
            "base": "base",
            "tuned": "tuned"
        }
        
        inference_model_type = model_type_map.get(request.model_type, "tuned")
        
        # Eğer tuned model yoksa base'e fallback
        if inference_model_type == "tuned" and classifier.tuned_model is None:
            logger.warning("⚠️ Tuned model yüklenmedi, base model kullanılıyor")
            inference_model_type = "base"
        
        # Sentiment analizi
        result = classifier.classify_sentiment(
            text=request.text,
            model_type=inference_model_type
        )
        
        # Hata kontrolü
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Bilinmeyen hata"))
        
        # Response'u API formatına dönüştür
        response_data = {
            "success": True,
            "text": result["text"],
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "model_used": request.model_type,
            "model_response": result.get("model_type", ""),
            "error": None
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Sentiment analiz hatası: {e}")
        raise HTTPException(status_code=500, detail="İç sunucu hatası")

@app.get("/models")
async def get_models():
    """Kullanılabilir modelleri listele"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modeller yüklenmedi")
    
    available_models = []
    
    # Base model her zaman mevcut
    available_models.append(
        ModelInfo(
            id="base",
            name="Base DistilBERT Model",
            description="Eğitilmemiş orijinal DistilBERT modeli",
            capabilities=["Genel sentiment analizi"],
            training_data=None
        )
    )
    
    # Tuned model kontrolü
    if classifier.tuned_model is not None:
        available_models.append(
            ModelInfo(
                id="tuned",
                name="Fine-Tuned DistilBERT Model", 
                description="LoRA ile IMDB sentiment analysis için fine-tuning edilmiş model",
                capabilities=["Film review analizi", "Yüksek doğruluklu sentiment classification"],
                training_data="IMDB 50K film incelemeleri"
            )
        )
    
    return {"available_models": available_models}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = classifier is not None and (
        classifier.base_model is not None or 
        classifier.tuned_model is not None
    )
    
    loaded_models = []
    if classifier:
        if classifier.base_model is not None:
            loaded_models.append("base")
        if classifier.tuned_model is not None:
            loaded_models.append("tuned")
    
    return HealthCheckResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        gpu_available=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        loaded_models=loaded_models
    )

@app.get("/test/sentiment")
async def test_sentiment():
    """Sentiment analysis test endpoint"""
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Modeller yüklenmedi")
        
        test_texts = [
            "This movie was absolutely fantastic with great acting and story!",
            "Terrible film, complete waste of time. Poor acting and boring storyline.",
            "The movie was okay, nothing special but not bad either."
        ]
        
        results = []
        for test_text in test_texts:
            result = classifier.classify_sentiment(
                text=test_text,
                model_type="tuned" if classifier.tuned_model else "base"
            )
            results.append({
                "text": test_text,
                "result": result
            })
        
        return {
            "test_cases": results,
            "models_available": {
                "base": classifier.base_model is not None,
                "tuned": classifier.tuned_model is not None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Model detayları"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modeller yüklenmedi")
    
    info = {}
    
    if classifier.base_model is not None:
        info["base"] = {
            "parameters": sum(p.numel() for p in classifier.base_model.parameters()),
            "device": str(next(classifier.base_model.parameters()).device),
            "model_type": type(classifier.base_model).__name__
        }
    
    if classifier.tuned_model is not None:
        info["tuned"] = {
            "parameters": sum(p.numel() for p in classifier.tuned_model.parameters()),
            "device": str(next(classifier.tuned_model.parameters()).device),
            "model_type": type(classifier.tuned_model).__name__
        }
    
    return info

if __name__ == "__main__":
    import uvicorn
    logger.info("🌐 DistilBERT Sentiment Analysis API sunucusu başlatılıyor...")
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        log_level="info"
    )
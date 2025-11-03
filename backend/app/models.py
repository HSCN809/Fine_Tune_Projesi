"""
Pydantic modelleri - Sadece Sentiment Analysis API şemaları
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class SentimentLabel(str, Enum):
    """Sentiment sınıfları - Sadece positive/negative"""
    POSITIVE = "positive"
    NEGATIVE = "negative"

class ModelType(str, Enum):
    """Model tipi seçenekleri - inference ile uyumlu"""
    BASE = "base"
    TRAINED = "tuned"

class SentimentRequest(BaseModel):
    """
    Sentiment analizi istek şeması
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Analiz edilecek metin (1-2000 karakter)",
        example="This movie was absolutely fantastic with great acting!"
    )
    
    model_type: ModelType = Field(
        default=ModelType.TRAINED,
        description="Kullanılacak model tipi: 'base' (orijinal) veya 'tuned' (fine-tuned)"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "The film had amazing cinematography but the plot was weak.",
                "model_type": "tuned"
            }
        }

class SentimentResponse(BaseModel):
    """
    Sentiment analizi yanıt şeması
    """
    success: bool = Field(description="İşlem başarılı mı?")
    text: str = Field(description="Analiz edilen orijinal metin")
    sentiment: SentimentLabel = Field(description="Tahmin edilen sentiment")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Tahmin güven skoru (0.0 - 1.0)"
    )
    model_used: ModelType = Field(description="Kullanılan model tipi")
    model_response: Optional[str] = Field(
        default=None,
        description="Modelin ham çıktısı (debug için)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Hata mesajı (success=false ise)"
    )

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "text": "This movie was absolutely fantastic with great acting!",
                "sentiment": "positive",
                "confidence": 0.92,
                "model_used": "tuned",
                "model_response": "positive"
            }
        }

class ModelInfo(BaseModel):
    """
    Model bilgisi şeması
    """
    id: ModelType = Field(description="Model ID")
    name: str = Field(description="Model adı")
    description: str = Field(description="Model açıklaması")
    capabilities: list[str] = Field(description="Model yetenekleri")
    training_data: Optional[str] = Field(default=None, description="Eğitim verisi")

class ModelsListResponse(BaseModel):
    """
    Model listesi yanıt şeması
    """
    available_models: list[ModelInfo] = Field(description="Kullanılabilir modeller")

class HealthCheckResponse(BaseModel):
    """
    Health check yanıt şeması
    """
    status: Literal["healthy", "unhealthy"] = Field(description="Sistem durumu")
    models_loaded: bool = Field(description="Modeller yüklü mü?")
    gpu_available: bool = Field(description="GPU kullanılabilir mi?")
    device: str = Field(description="Kullanılan cihaz (cuda/cpu)")
    loaded_models: list[str] = Field(description="Yüklenen model listesi")

class ErrorResponse(BaseModel):
    """
    Hata yanıt şeması
    """
    success: bool = Field(default=False, description="İşlem başarısız")
    error: str = Field(description="Hata mesajı")
    detail: Optional[str] = Field(default=None, description="Detaylı hata bilgisi")

# Test fonksiyonu
def test_models():
    """Model sınıflarını test et"""
    print("🧪 Sentiment modeller test ediliyor...")
    
    # SentimentRequest test
    sentiment_req = SentimentRequest(
        text="This movie was great!",
        model_type="tuned"
    )
    assert sentiment_req.text == "This movie was great!"
    assert sentiment_req.model_type == "tuned"
    
    # SentimentResponse test
    sentiment_resp = SentimentResponse(
        success=True,
        text="Great movie!",
        sentiment="positive",
        confidence=0.9,
        model_used="tuned",
        model_response="positive"
    )
    assert sentiment_resp.success == True
    assert sentiment_resp.sentiment == "positive"
    assert sentiment_resp.confidence == 0.9
    
    # ModelInfo test
    model_info = ModelInfo(
        id="tuned",
        name="Fine-Tuned Model",
        description="LoRA ile fine-tuning edilmiş model",
        capabilities=["Film review analizi", "Sentiment classification"],
        training_data="IMDB 50K reviews"
    )
    assert model_info.id == "tuned"
    assert len(model_info.capabilities) == 2
    
    print("✅ Sentiment modeller testi başarılı!")

if __name__ == "__main__":
    test_models()
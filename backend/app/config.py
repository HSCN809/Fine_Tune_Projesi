"""
Konfigürasyon ayarları - DistilBERT için güncellendi
"""
import os
from typing import Dict
from pathlib import Path

# Path'leri dinamik olarak ayarla
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
models_dir = project_root / "models"

class Settings:
    """Uygulama ayarları - DistilBERT için güncellendi"""
    
    # Model ayarları - DistilBERT
    BASE_MODEL_NAME: str = "distilbert-base-uncased"
    LORA_MODEL_PATH: str = str(models_dir / "lora_imdb_classification")
    
    # API ayarları
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # CORS ayarları
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Model parametreleri
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 8
    
    # Classification ayarları
    CLASS_NAMES: Dict[int, str] = {0: "negative", 1: "positive"}
    CONFIDENCE_THRESHOLD: float = 0.5

# Global settings
settings = Settings()

def test_config():
    """Konfigürasyon testi"""
    print("🧪 Config test...")
    print(f"Base Model: {settings.BASE_MODEL_NAME}")
    print(f"LoRA Path: {settings.LORA_MODEL_PATH}")
    print(f"Path exists: {os.path.exists(settings.LORA_MODEL_PATH)}")
    print(f"API Port: {settings.API_PORT}")
    print(f"Max Length: {settings.MAX_LENGTH}")
    print(f"Class Names: {settings.CLASS_NAMES}")
    print("✅ Config test tamamlandı")

if __name__ == "__main__":
    test_config()
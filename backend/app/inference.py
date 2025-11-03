"""
SEQUENCE CLASSIFICATION INFERENCE - CLEAN VERSION
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistilBERTSentimentClassifier:
    """DistilBERT ile sentiment sınıflandırma - CLEAN VERSION"""
    
    def __init__(self, model_path="distilbert-base-uncased", lora_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎯 Cihaz: {self.device}")
        
        # DYNAMIC PATH ÇÖZÜMÜ
        if lora_path and not os.path.isabs(lora_path):
            current_dir = Path(__file__).parent
            lora_path = current_dir / lora_path
            logger.info(f"📁 Fine-tuned model path: {lora_path}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # BASE model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1}
        )
        self.base_model.to(self.device)
        self.base_model.eval()
        logger.info("✅ Base DistilBERT yüklendi")
        
        # FINE-TUNED model
        self.tuned_model = None
        if lora_path and os.path.exists(lora_path):
            try:
                logger.info(f"🔄 Fine-tuned model yükleniyor: {lora_path}")
                
                self.tuned_model = AutoModelForSequenceClassification.from_pretrained(
                    str(lora_path),
                    id2label={0: "negative", 1: "positive"},
                    label2id={"negative": 0, "positive": 1}
                )
                self.tuned_model.to(self.device)
                self.tuned_model.eval()
                logger.info("✅ Fine-tuned DistilBERT yüklendi!")
                
            except Exception as e:
                logger.error(f"❌ Fine-tuned model yüklenemedi: {e}")
                self.tuned_model = None
        else:
            if lora_path:
                logger.warning(f"⚠️ Fine-tuned model path bulunamadı: {lora_path}")
    
    def classify_sentiment(self, text, model_type="tuned"):
        """SEQUENCE CLASSIFICATION ile sentiment analizi"""
        try:
            if model_type == "tuned" and self.tuned_model is not None:
                model = self.tuned_model
                model_name = "FINE-TUNED"
            else:
                model = self.base_model
                model_name = "BASE"
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
                
                sentiment = model.config.id2label[predicted_class]
                
                if predicted_class == 1:
                    positive_prob = float(probabilities[0][1])
                    negative_prob = float(probabilities[0][0])
                else:
                    positive_prob = float(probabilities[0][1])
                    negative_prob = float(probabilities[0][0])
            
            logger.info(f"🎯 {model_name}: {sentiment.upper()} ({confidence:.3f})")
            
            return {
                "success": True,
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "model_type": model_name,
                "probabilities": {
                    "positive": positive_prob,
                    "negative": negative_prob
                }
            }
            
        except Exception as e:
            logger.error(f"❌ {model_type} model sınıflandırma hatası: {e}")
            return {"success": False, "error": str(e), "model_type": model_type}

# DYNAMIC PATH AYARLARI
def setup_paths():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    models_dir = project_root / "models"
    lora_path = models_dir / "lora_imdb_classification"
    
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"📁 Models directory: {models_dir}")
    logger.info(f"📁 Fine-tuned model path: {lora_path}")
    
    return str(lora_path)

# KULLANIM
lora_path = setup_paths()
classifier = DistilBERTSentimentClassifier(lora_path=lora_path)

# TEST METİNLERİ
test_texts = [
    "This movie is absolutely fantastic! The acting was superb and cinematography breathtaking.",
    "Terrible film, complete waste of time. Poor acting and boring storyline.",
    "The acting was good but the story was weak and predictable.",
    "A masterpiece! One of the best films I've ever seen in my life.",
    "Boring and pointless. I fell asleep halfway through the movie.",
    "Amazing performance by the lead actor, but the script could be better.",
    "I loved every moment of this film, it was perfect from start to finish.",
    "Worst movie ever made, terrible acting and awful plot.",
    "The cinematography was beautiful but the characters were poorly developed.",
    "An incredible experience that left me speechless and wanting more."
]

print("🧠 DISTILBERT SENTIMENT ANALYSIS")
print("=" * 70)

for i, text in enumerate(test_texts, 1):
    print(f"\n{i}. METİN: {text}")
    print("-" * 70)
    
    # FINE-TUNED model
    if classifier.tuned_model:
        tuned_result = classifier.classify_sentiment(text, model_type="tuned")
        if tuned_result["success"]:
            print(f"   ✅ FINE-TUNED: {tuned_result['sentiment'].upper()} (Confidence: {tuned_result['confidence']:.3f})")
            print(f"      📊 Positive: {tuned_result['probabilities']['positive']:.3f}, Negative: {tuned_result['probabilities']['negative']:.3f}")
    
    # BASE model
    base_result = classifier.classify_sentiment(text, model_type="base")
    if base_result["success"]:
        print(f"   🔵 BASE: {base_result['sentiment'].upper()} (Confidence: {base_result['confidence']:.3f})")
        print(f"      📊 Positive: {base_result['probabilities']['positive']:.3f}, Negative: {base_result['probabilities']['negative']:.3f}")
    
    # KARŞILAŞTIRMA
    if classifier.tuned_model and tuned_result["success"] and base_result["success"]:
        if tuned_result["sentiment"] == base_result["sentiment"]:
            confidence_diff = tuned_result["confidence"] - base_result["confidence"]
            print(f"   🔄 AYNI SONUÇ | Confidence Farkı: {confidence_diff:+.3f}")
        else:
            print(f"   ⚡ FARKLI SONUÇ | Tuned: {tuned_result['sentiment']}, Base: {base_result['sentiment']}")
    
    print("=" * 70)

# HIZ TESTİ
def performance_comparison():
    print(f"\n⚡ PERFORMANCE COMPARISON")
    print("=" * 40)
    
    test_text = "This is a wonderful and amazing product that I truly love!"
    
    models_to_test = []
    if classifier.tuned_model:
        models_to_test.append(("FINE-TUNED", "tuned"))
    models_to_test.append(("BASE", "base"))
    
    for model_name, model_type in models_to_test:
        print(f"\n🔍 {model_name} MODEL:")
        
        start_time = time.time()
        results = []
        
        for i in range(5):
            result = classifier.classify_sentiment(f"{test_text} - Test {i+1}", model_type)
            results.append(result)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        
        print(f"   ⏱️  Average inference time: {avg_time:.3f}s")
        
        if results[0]["success"]:
            print(f"   🎯 Sample result: {results[0]['sentiment']} ({results[0]['confidence']:.3f})")

if __name__ == "__main__":
    performance_comparison()
    
    print(f"\n📋 MODEL STATUS:")
    print(f"   ✅ Base Model: Loaded")
    print(f"   {'✅' if classifier.tuned_model else '❌'} Fine-tuned Model: {'Loaded' if classifier.tuned_model else 'Not loaded'}")
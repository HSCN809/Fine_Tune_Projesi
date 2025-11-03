"""
LoRA ile IMDB SEQUENCE CLASSIFICATION - GPU DESTEKLİ (FP16 SCALER FIX)
"""
import os
import sys
import logging
from pathlib import Path

# Root directory'yi Python path'ine ekle
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, concatenate_datasets
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_device():
    """GPU/CPU ayarını yap"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🎮 GPU kullanılacak: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        logger.warning("⚠️ GPU bulunamadı, CPU kullanılacak")
    
    return device

def setup_model_and_tokenizer(model_name="distilbert-base-uncased"):
    """SEQUENCE CLASSIFICATION için model ve tokenizer - SCALER FIX"""
    logger.info(f"📦 Classification model yükleniyor: {model_name}")
    
    device = setup_device()
    
    # 🎯 SEQUENCE CLASSIFICATION MODEL - SCALER FIX
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
        # 🚨 torch_dtype KALDIRILDI - scaler hatası veriyor
    )
    
    # Pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    logger.info(f"✅ Classification model yüklendi: {model.device}")
    return model, tokenizer, device

def setup_lora_for_classification(model):
    """SEQUENCE CLASSIFICATION için LoRA konfigürasyonu"""
    logger.info("🎯 Classification için LoRA ayarlanıyor...")
    
    try:
        # DistilBERT için doğru modüller
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        logger.info("✅ Classification LoRA başarıyla ayarlandı")
        return model
        
    except Exception as e:
        logger.error(f"❌ LoRA hatası: {e}")
        raise

def compute_metrics(eval_pred):
    """Classification metrics hesaplama"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_user_input():
    """Kullanıcıdan train/test miktarlarını al"""
    print("\n" + "="*50)
    print("🎯 TRAIN/TEST VERİ MİKTARLARINI AYARLAYIN")
    print("="*50)
    
    dataset = load_dataset("imdb")
    total_samples = len(dataset['train']) + len(dataset['test'])
    print(f"📊 IMDB'de toplam {total_samples} örnek bulunuyor")
    
    while True:
        try:
            print("\n💡 Öneriler:")
            print("   - Hızlı test için: 1000 train, 200 test")
            print("   - Orta ölçek için: 10000 train, 2000 test") 
            print("   - Full dataset için: 40000 train, 10000 test")
            
            train_samples = int(input("\n🟢 Kaç tane TRAIN örneği kullanılsın? : "))
            test_samples = int(input("🔴 Kaç tane TEST örneği kullanılsın? : "))
            
            if train_samples <= 0 or test_samples <= 0:
                print("❌ Lütfen pozitif sayı girin!")
                continue
                
            if train_samples + test_samples > total_samples:
                print(f"❌ Toplam {train_samples + test_samples} örnek istediniz ama sadece {total_samples} mevcut!")
                continue
                
            total = train_samples + test_samples
            train_ratio = (train_samples / total) * 100
            test_ratio = (test_samples / total) * 100
            
            print(f"\n📈 Seçilen dağılım:")
            print(f"   → Train: {train_samples} örnek (%{train_ratio:.1f})")
            print(f"   → Test:  {test_samples} örnek (%{test_ratio:.1f})")
            print(f"   → Toplam: {total} örnek")
            
            confirm = input("\n✅ Bu ayarlarla devam etmek istiyor musunuz? (y/n): ")
            if confirm.lower() == 'y':
                return train_samples, test_samples
            else:
                print("🔄 Ayarlar sıfırlandı, tekrar deneyin...")
                
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin!")
        except KeyboardInterrupt:
            print("\n⏹️ İşlem iptal edildi")
            sys.exit(0)

def prepare_classification_data(tokenizer, train_samples, test_samples, max_length=256):
    """
    SEQUENCE CLASSIFICATION için veri hazırlama
    """
    logger.info(f"📊 Classification verisi hazırlanıyor: {train_samples} train, {test_samples} test")
    
    try:
        # IMDB datasetini yükle
        dataset = load_dataset("imdb")
        logger.info(f"✅ IMDB yüklendi: {len(dataset['train'])} train, {len(dataset['test'])} test")
        
        # Dataset'leri birleştir
        full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
        logger.info(f"📦 Toplam veri: {len(full_dataset)} örnek")
        
        # Rastgele shuffle yap
        full_dataset = full_dataset.shuffle(seed=42)
        logger.info("🔀 Veri karıştırıldı")
        
        # Kullanıcının istediği kadar veri al
        total_needed = train_samples + test_samples
        selected_data = full_dataset.select(range(total_needed))
        
        # Train/test split yap
        split_dataset = selected_data.train_test_split(
            test_size=test_samples,
            shuffle=True,
            seed=42
        )
        
        logger.info(f"🎯 Split tamamlandı: {len(split_dataset['train'])} train, {len(split_dataset['test'])} test")
        
        def tokenize_function(examples):
            """🎯 SADECE SINIFLANDIRMA İÇİN TOKENIZATION"""
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors=None,
            )
        
        # Tokenize et
        tokenized_train = split_dataset["train"].map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']
        )
        tokenized_test = split_dataset["test"].map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']
        )
        
        logger.info("✅ Classification verisi hazırlandı")
        
        return tokenized_train, tokenized_test
        
    except Exception as e:
        logger.error(f"❌ Veri hazırlama hatası: {e}")
        raise

def train_classification_model(train_samples, test_samples):
    """SEQUENCE CLASSIFICATION fine-tuning - SCALER FIX"""
    logger.info(f"🚀 LoRA + IMDB Classification Başlıyor... ({train_samples} train, {test_samples} test)")
    
    try:
        # 1. Classification model ve tokenizer
        model, tokenizer, device = setup_model_and_tokenizer()
        
        # 2. Classification için LoRA setup
        model = setup_lora_for_classification(model)
        
        # 3. Classification verisini hazırla
        train_dataset, eval_dataset = prepare_classification_data(tokenizer, train_samples, test_samples)
        
        # 4. Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # 5. Training arguments - OPTIMIZED FOR RTX 5070 🎯
        training_args = TrainingArguments(
            output_dir="../models/lora_imdb_classification",
            num_train_epochs=3,
            per_device_train_batch_size=8,  # 🎯 Optimized batch size
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            warmup_steps=50,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-4,
            # 🎯 SCALER FIX - FP16 KAPALI, BF16 AKTIF
            fp16=False,  # 🚨 FP16 KAPALI - scaler hatası veriyor
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # 🎯 BF16 deneniyor
            half_precision_backend="auto",
            remove_unused_columns=True,
            report_to=None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            max_grad_norm=1.0,
            eval_accumulation_steps=1,
            save_total_limit=2,
            logging_dir="./logs",
            # 🎯 OPTIMIZER AYARLARI
            optim="adamw_torch",  # 🎯 AdamW optimizer
            weight_decay=0.01,
            label_smoothing_factor=0.1
        )
        
        # EĞİTİM DETAYLARI
        total_train_samples = len(train_dataset)
        total_eval_samples = len(eval_dataset)
        
        logger.info("📊 CLASSIFICATION EĞİTİM DETAYLARI:")
        logger.info(f"   → Cihaz: {device.upper()}")
        logger.info(f"   → FP16: False (scaler hatası nedeniyle kapalı)")
        logger.info(f"   → BF16: {training_args.bf16}")
        logger.info(f"   → Train örnekleri: {total_train_samples}")
        logger.info(f"   → Test örnekleri: {total_eval_samples}")
        logger.info(f"   → Epoch sayısı: {training_args.num_train_epochs}")
        logger.info(f"   → Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"   → Learning rate: {training_args.learning_rate}")
        logger.info(f"   → Optimizer: {training_args.optim}")
        
        # 6. Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # 7. Eğitim
        logger.info("🎯 Classification eğitimi başlıyor...")
        train_result = trainer.train()
        
        # 8. Modeli kaydet
        logger.info("💾 Model kaydediliyor...")
        trainer.save_model()
        tokenizer.save_pretrained("../models/lora_imdb_classification")
        
        # Final evaluation
        logger.info("📊 Final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"🎯 Final metrics: {eval_results}")
        
        # Metrics
        metrics = train_result.metrics
        logger.info(f"📊 Eğitim tamamlandı: {metrics}")
        
        training_time = metrics.get('train_runtime', 0)
        logger.info(f"⏱️  Toplam eğitim süresi: {training_time:.2f} saniye ({training_time/60:.2f} dakika)")
        
        logger.info("✅ Classification model başarıyla kaydedildi: ../models/lora_imdb_classification")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Eğitim hatası: {e}")
        # Detaylı hata mesajı
        import traceback
        logger.error(f"❌ Detaylı hata: {traceback.format_exc()}")
        
        # Alternatif: FP32 ile dene
        logger.info("🔄 FP32 ile deneniyor...")
        try:
            # FP32 fallback
            training_args.fp16 = False
            training_args.bf16 = False
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            train_result = trainer.train()
            trainer.save_model()
            logger.info("✅ FP32 ile eğitim başarılı!")
            return trainer
            
        except Exception as e2:
            logger.error(f"❌ FP32 de başarısız: {e2}")
            raise

if __name__ == "__main__":
    print("=" * 60)
    print("🎬 IMDB SEQUENCE CLASSIFICATION - SCALER FIX")
    print("=" * 60)
    
    try:
        # Kullanıcıdan veri miktarlarını al
        train_samples, test_samples = get_user_input()
        
        # Onay
        user_input = input("\n🚀 Classification eğitimine başlamak istiyor musunuz? (y/n): ")
        if user_input.lower() == 'y':
            print("\n🔥 CLASSIFICATION EĞİTİMİ BAŞLIYOR...")
            train_classification_model(train_samples, test_samples)
        else:
            print("ℹ️ Eğitim iptal edildi.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Program kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
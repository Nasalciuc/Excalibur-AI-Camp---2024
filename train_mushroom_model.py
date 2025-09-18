"""
Script de antrenare optimizat pentru RTX 4050 (6GB VRAM)
Dataset: Mushroom Identifier (chanterelle, death-cap, field-mushroom)
"""

from ultralytics import YOLO
import torch
import gc
import os

def check_gpu_memory():
    """Verifică memoria GPU disponibilă"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        free = total_memory - cached
        
        print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
        print(f"📊 Total VRAM: {total_memory:.1f}GB")
        print(f"🔄 Allocated: {allocated:.1f}GB")
        print(f"💾 Cached: {cached:.1f}GB") 
        print(f"✅ Free: {free:.1f}GB")
        print("-" * 40)
        
        return free > 2.0  # Avem nevoie de cel puțin 2GB liberi
    return False

def optimize_for_rtx4050():
    """Optimizări specifice pentru RTX 4050"""
    # Curăță cache-ul GPU
    torch.cuda.empty_cache()
    gc.collect()
    
    # Setări pentru eficiență memorie
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("🔧 Optimizări aplicate pentru RTX 4050")

def train_mushroom_detector():
    """Antrenează modelul de detecție ciuperci"""
    
    # Verifică GPU
    if not check_gpu_memory():
        print("⚠️  Memorie GPU insuficientă!")
        return
    
    # Aplică optimizări
    optimize_for_rtx4050()
    
    # Încarcă modelul cel mai mic (nano)
    print("📥 Încărcare model YOLOv11 nano...")
    model = YOLO('yolo11n.pt')  # Modelul cel mai mic pentru 6GB VRAM
    
    # Parametri optimizați pentru RTX 4050
    training_params = {
        'data': 'data.yaml',           # Dataset path
        'epochs': 100,                 # Numărul de epoci
        'imgsz': 480,                  # Dimensiune mai mică (în loc de 640)
        'batch': 4,                    # Batch size foarte mic pentru 6GB
        'patience': 15,                # Early stopping
        'save': True,                  # Salvează modelul
        'device': 0,                   # Prima GPU
        'workers': 2,                  # Mai puține procese worker
        'amp': True,                   # Mixed precision (economisește VRAM)
        'cache': False,                # Nu cache datele în RAM
        'close_mosaic': 15,            # Dezactivează mosaic în ultimele epoci
        'name': 'mushroom_detector_rtx4050',
        'project': 'runs/detect',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',          # Optimizer eficient
        'lr0': 0.001,                  # Learning rate mai mic
        'weight_decay': 0.0005,        # Regularizare
        'warmup_epochs': 3,            # Warmup pentru stabilitate
        'box': 7.5,                    # Loss weight pentru bounding boxes
        'cls': 0.5,                    # Loss weight pentru clasificare
        'dfl': 1.5,                    # Distribution focal loss weight
        'hsv_h': 0.015,                # Augmentare culoare
        'hsv_s': 0.7,                  # Augmentare saturație
        'hsv_v': 0.4,                  # Augmentare luminozitate
        'degrees': 0.0,                # Fără rotație (economisește compute)
        'translate': 0.1,              # Translație minimă
        'scale': 0.5,                  # Scaling augmentation
        'shear': 0.0,                  # Fără shear
        'perspective': 0.0,            # Fără perspective
        'flipud': 0.0,                 # Fără flip vertical
        'fliplr': 0.5,                 # Flip orizontal
        'mosaic': 1.0,                 # Mosaic augmentation
        'mixup': 0.0,                  # Fără mixup (economisește VRAM)
        'copy_paste': 0.0,             # Fără copy-paste
    }
    
    print("🚀 Începe antrenarea cu parametri optimizați...")
    print(f"📋 Parametri cheie:")
    print(f"   • Batch size: {training_params['batch']}")
    print(f"   • Image size: {training_params['imgsz']}")
    print(f"   • Mixed precision: {training_params['amp']}")
    print(f"   • Workers: {training_params['workers']}")
    print("-" * 40)
    
    try:
        # Antrenează modelul
        results = model.train(**training_params)
        
        print("✅ Antrenare completă!")
        print(f"📁 Modelul salvat în: runs/detect/mushroom_detector_rtx4050/")
        
        # Validare finală
        print("🧪 Rulare validare finală...")
        metrics = model.val()
        
        print(f"📊 Rezultate finale:")
        print(f"   • mAP50: {metrics.box.map50:.3f}")
        print(f"   • mAP50-95: {metrics.box.map:.3f}")
        print(f"   • Precision: {metrics.box.mp:.3f}")
        print(f"   • Recall: {metrics.box.mr:.3f}")
        
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ CUDA out of memory!")
            print("💡 Încercați să reduceți batch_size la 2 sau 1")
            print("💡 Sau reduceți imgsz la 416")
        else:
            print(f"❌ Eroare: {e}")
        return None
    
    finally:
        # Curăță memoria
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    print("🍄 Antrenare Model Detecție Ciuperci")
    print("🎯 Dataset: chanterelle, death-cap, field-mushroom")
    print("🖥️  Optimizat pentru RTX 4050 (6GB VRAM)")
    print("=" * 50)
    
    results = train_mushroom_detector()
    
    if results:
        print("\n🎉 Antrenare reușită!")
        print("📝 Pentru a testa modelul:")
        print("   python test_model.py")
    else:
        print("\n❌ Antrenarea a eșuat!")
        print("💡 Verificați memoria GPU și reduceți parametrii")
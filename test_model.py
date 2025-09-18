"""
Script pentru testarea modelului antrenat de detecție ciuperci
Optimizat pentru RTX 4050
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def test_trained_model():
    """Testează modelul antrenat pe imagini de test"""
    
    # Calea către modelul antrenat
    model_path = "runs/detect/mushroom_detector_rtx4050/weights/best.pt"
    
    if not Path(model_path).exists():
        print("❌ Modelul antrenat nu a fost găsit!")
        print(f"🔍 Căutați în: {model_path}")
        return
    
    # Încarcă modelul antrenat
    print("📥 Încărcare model antrenat...")
    model = YOLO(model_path)
    
    # Definește tipurile de ciuperci
    class_names = {
        0: "chanterelle (galbiori) ✅",
        1: "death-cap (coprini) ⚠️ TOXIC!",
        2: "field-mushroom (de câmp) ✅"
    }
    
    # Testează pe câteva imagini din setul de test
    test_dir = Path("test/images")
    test_images = list(test_dir.glob("*.jpg"))[:5]  # Primele 5 imagini
    
    print(f"🧪 Testez pe {len(test_images)} imagini...")
    
    for img_path in test_images:
        print(f"\n📸 Procesez: {img_path.name}")
        
        # Predicție
        results = model.predict(
            source=str(img_path),
            save=True,
            conf=0.25,  # Confidence threshold
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        # Afișează rezultatele
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    mushroom_type = class_names.get(cls, "necunoscut")
                    
                    print(f"   🍄 Detectat: {mushroom_type}")
                    print(f"   📊 Confidence: {conf:.2%}")
                    
                    # Avertizare pentru ciuperci toxice
                    if cls == 1:  # death-cap
                        print("   ⚠️  ATENȚIE: CIUPERCĂ TOXICĂ!")
            else:
                print("   ❓ Nu s-au detectat ciuperci")
    
    print(f"\n✅ Test complet! Rezultatele salvate în: runs/detect/predict/")

def run_interactive_test():
    """Test interactiv pe o imagine specificată"""
    
    model_path = "runs/detect/mushroom_detector_rtx4050/weights/best.pt"
    
    if not Path(model_path).exists():
        print("❌ Rulați mai întâi train_mushroom_model.py")
        return
    
    model = YOLO(model_path)
    
    # Cere utilizatorului să specifice o imagine
    img_path = input("📁 Introduceți calea către imaginea de test: ").strip()
    
    if not Path(img_path).exists():
        print("❌ Imaginea nu există!")
        return
    
    print("🔍 Analizez imaginea...")
    
    # Predicție
    results = model.predict(
        source=img_path,
        save=True,
        show=True,  # Afișează rezultatul
        conf=0.25
    )
    
    # Interpretează rezultatele
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"\n🍄 Găsite {len(boxes)} ciuperci:")
            for i, box in enumerate(boxes, 1):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                mushroom_types = {
                    0: ("Chanterelle (Galbiori)", "✅ COMESTIBIL", "🟢"),
                    1: ("Death-cap (Coprini)", "⚠️ EXTREM DE TOXIC!", "🔴"),
                    2: ("Field Mushroom (De câmp)", "✅ COMESTIBIL", "🟢")
                }
                
                name, safety, emoji = mushroom_types.get(cls, ("Necunoscut", "❓ NEIDENTIFICAT", "⚪"))
                
                print(f"   {i}. {emoji} {name}")
                print(f"      Siguranță: {safety}")
                print(f"      Confidence: {conf:.2%}")
                
                if cls == 1:
                    print("      🚨 NU CONSUMAȚI! Contactați un specialist!")
        else:
            print("❓ Nu s-au detectat ciuperci în imagine")

if __name__ == "__main__":
    print("🍄 Test Model Detecție Ciuperci")
    print("=" * 40)
    
    choice = input("Alegeți modul de test:\n1. Test automat pe imagini\n2. Test interactiv\nOpțiune (1/2): ").strip()
    
    if choice == "1":
        test_trained_model()
    elif choice == "2":
        run_interactive_test()
    else:
        print("❌ Opțiune invalidă!")
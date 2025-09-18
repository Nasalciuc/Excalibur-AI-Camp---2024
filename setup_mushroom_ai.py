"""
🍄 Mushroom Identifier - Setup Automat
Instalare automată a dependințelor și antrenare model
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Execută o comandă și afișează progresul"""
    print(f"\n🔄 {description}...")
    print(f"Comanda: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - Succes!")
        if result.stdout:
            print(f"Output: {result.stdout[:500]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Eșuat!")
        print(f"Error: {e.stderr}")
        return False

def check_gpu():
    """Verifică disponibilitatea GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detectat: {gpu_name}")
            return True
        else:
            print("⚠️  Nu a fost detectat GPU CUDA - training va fi lent")
            return False
    except ImportError:
        print("❌ PyTorch nu este instalat")
        return False

def main():
    print("🍄 MUSHROOM IDENTIFIER - SETUP AUTOMAT")
    print("=" * 50)
    
    # Verifică Python
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 8:
        print("❌ Necesită Python 3.8 sau mai nou!")
        return False
    
    # Pasul 1: Instalare PyTorch cu CUDA
    print("\n📦 PASUL 1: Instalare PyTorch cu CUDA")
    torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    if not run_command(torch_command, "Instalare PyTorch cu CUDA"):
        print("⚠️  Încerc instalarea CPU only...")
        cpu_command = "pip install torch torchvision torchaudio"
        run_command(cpu_command, "Instalare PyTorch CPU")
    
    # Pasul 2: Instalare Ultralytics și dependințe
    print("\n📦 PASUL 2: Instalare YOLOv11 și dependințe")
    deps_command = "pip install ultralytics opencv-python pyyaml pandas seaborn matplotlib"
    run_command(deps_command, "Instalare dependințe")
    
    # Pasul 3: Verificare setup
    print("\n🔍 PASUL 3: Verificare configurație")
    if os.path.exists("pre_training_check.py"):
        run_command("python pre_training_check.py", "Verificare setup complet")
    else:
        print("⚠️  pre_training_check.py nu a fost găsit")
    
    # Verifică GPU
    gpu_available = check_gpu()
    
    # Pasul 4: Opțiuni de antrenare
    print("\n🚀 PASUL 4: Opțiuni de continuare")
    print("Alege o opțiune:")
    print("1. Antrenează modelul acum (recomandat)")
    print("2. Doar setup - antrenez mai târziu")
    print("3. Test rapid pe imagini existente")
    
    choice = input("\nIntroduci opțiunea (1-3): ").strip()
    
    if choice == "1":
        print("\n🎯 Începe antrenarea...")
        if gpu_available:
            print("⚡ Training pe GPU - estimat 45-60 minute")
        else:
            print("🐌 Training pe CPU - estimat 3-5 ore")
        
        confirm = input("Continui? (y/n): ").lower()
        if confirm == 'y':
            if os.path.exists("train_mushroom_model.py"):
                run_command("python train_mushroom_model.py", "Antrenare model mushroom")
            else:
                print("❌ train_mushroom_model.py nu a fost găsit")
    
    elif choice == "2":
        print("✅ Setup complet! Pentru antrenare rulează:")
        print("   python train_mushroom_model.py")
    
    elif choice == "3":
        print("🧪 Test rapid...")
        if os.path.exists("test_model.py"):
            # Verifică dacă există model antrenat
            model_path = Path("runs/detect/mushroom_detector_rtx4050/weights/best.pt")
            if model_path.exists():
                run_command("python test_model.py --auto", "Test pe imagini")
            else:
                print("❌ Nu există model antrenat. Rulează întâi training.")
        else:
            print("❌ test_model.py nu a fost găsit")
    
    print("\n🎉 Setup finalizat!")
    print("\n📖 Pentru mai multe informații:")
    print("   - README.md - Ghid complet")
    print("   - TRAINING_RESULTS.md - Rezultate antrenare")
    print("   - MODEL_DOWNLOAD.md - Ghid model")

if __name__ == "__main__":
    main()
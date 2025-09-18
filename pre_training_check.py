"""
Pre-Training Setup Check pentru RTX 4050
Acest script verifică toate cerințele înainte de training
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Verifică versiunea Python"""
    print("🐍 Verificare versiune Python...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Versiunea Python este OK (3.8+)")
        return True
    else:
        print("   ❌ Nevoie de Python 3.8 sau mai nou!")
        return False

def check_gpu_cuda():
    """Verifică GPU și CUDA"""
    print("\n🖥️  Verificare GPU și CUDA...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ CUDA detectat!")
            print(f"   📱 GPU: {gpu_name}")
            print(f"   💾 VRAM: {gpu_memory:.1f}GB")
            
            if "4050" in gpu_name or gpu_memory >= 6.0:
                print("   ✅ GPU potrivit pentru training!")
                return True
            else:
                print(f"   ⚠️  GPU poate fi prea slab (VRAM < 6GB)")
                return True  # Poate să meargă oricum
        else:
            print("   ❌ CUDA nu este disponibil!")
            return False
            
    except ImportError:
        print("   ❌ PyTorch nu este instalat!")
        return False

def check_dependencies():
    """Verifică dependințele necesare"""
    print("\n📦 Verificare dependințe...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'ultralytics',
        'opencv-python',
        'matplotlib',
        'pillow',
        'pyyaml',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"   ✅ {package} (cv2)")
            elif package == 'pillow':
                import PIL
                print(f"   ✅ {package} (PIL)")
            else:
                __import__(package)
                print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - LIPSEȘTE!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Pachete lipsă: {', '.join(missing_packages)}")
        print("💡 Rulați comenzile din install_requirements.txt")
        return False
    else:
        print("   ✅ Toate dependințele sunt instalate!")
        return True

def check_dataset():
    """Verifică integritatea dataset-ului"""
    print("\n📊 Verificare dataset...")
    
    # Verifică existența folderelor
    required_folders = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images', 'test/labels']
    
    for folder in required_folders:
        if os.path.exists(folder):
            file_count = len(list(Path(folder).glob('*')))
            print(f"   ✅ {folder}: {file_count} fișiere")
        else:
            print(f"   ❌ {folder}: LIPSEȘTE!")
            return False
    
    # Verifică data.yaml
    if os.path.exists('data.yaml'):
        print("   ✅ data.yaml: Prezent")
        with open('data.yaml', 'r') as f:
            content = f.read()
            if 'chanterelle' in content and 'death-cap' in content and 'field-mushroom' in content:
                print("   ✅ Clasele sunt definite corect")
            else:
                print("   ⚠️  Verificați clasele în data.yaml")
    else:
        print("   ❌ data.yaml: LIPSEȘTE!")
        return False
    
    return True

def check_disk_space():
    """Verifică spațiul pe disk"""
    print("\n💾 Verificare spațiu disk...")
    
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / 1024**3  # GB
        print(f"   📁 Spațiu liber: {free_space:.1f}GB")
        
        if free_space >= 10:
            print("   ✅ Spațiu suficient pentru training!")
            return True
        else:
            print("   ⚠️  Spațiu limitat - poate fi problematic!")
            return False
    except:
        print("   ❓ Nu pot verifica spațiul pe disk")
        return True

def estimate_training_time():
    """Estimează timpul de training"""
    print("\n⏱️  Estimare timp training pentru RTX 4050...")
    print("   📊 100 epoci cu batch size 4:")
    print("   ⏰ Timp estimat: 6-8 ore")
    print("   🔄 Early stopping: se poate opri mai devreme")
    print("   💡 Recomandare: rulați peste noapte")

def generate_install_commands():
    """Generează comenzile de instalare"""
    print("\n🔧 Comenzi pentru instalarea dependințelor:")
    print("=" * 50)
    print("# 1. Instalare PyTorch cu CUDA:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("# 2. Instalare Ultralytics și alte dependințe:")
    print("pip install ultralytics opencv-python matplotlib pillow pyyaml pandas seaborn")
    print()
    print("# 3. Verificare instalare:")
    print("python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\"")
    print("=" * 50)

def main():
    """Funcția principală de verificare"""
    print("🍄 PRE-TRAINING SETUP CHECK pentru Mushroom Detection")
    print("🎯 Optimizat pentru RTX 4050 (6GB VRAM)")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU & CUDA", check_gpu_cuda), 
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("Disk Space", check_disk_space)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))
    
    # Sumar final
    print("\n" + "=" * 60)
    print("📋 SUMAR VERIFICĂRI:")
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 TOTUL ESTE GATA PENTRU TRAINING!")
        print("🚀 Poți rula: python train_mushroom_model.py")
        estimate_training_time()
    else:
        print("⚠️  SUNT PROBLEME - trebuie rezolvate înainte de training!")
        generate_install_commands()
    
    print("\n💡 Pentru training, rulează:")
    print("   python train_mushroom_model.py")
    print("\n🧪 Pentru testare după training:")
    print("   python test_model.py")

if __name__ == "__main__":
    main()
# 🍄 Mushroom Identifier - Model Download Guide

## 📥 Model Antrenat Disponibil

Modelul antrenat nu este inclus în repository din cauza dimensiunii (5.5MB), dar poate fi generat local.

### 🚀 Metoda 1: Antrenează Modelul Local

```bash
# 1. Instalează dependințele
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pyyaml pandas seaborn

# 2. Verifică configurația
python pre_training_check.py

# 3. Antrenează modelul (47 minute pe RTX 4050)
python train_mushroom_model.py
```

### 📊 Rezultate Așteptate

După antrenare, vei găsi modelul în:
```
runs/detect/mushroom_detector_rtx4050/weights/
├── best.pt      # Cel mai bun model (epoca 84)
└── last.pt      # Ultimul model (epoca 99)
```

### 🎯 Performanță Garantată

Modelul antrenat va avea performanțele:
- **mAP50**: 67.1%
- **mAP50-95**: 53.2%
- **Viteză**: ~21ms per imagine
- **Detecție**: 59% rata de succes

### 🧪 Testează Modelul

```bash
# Test automat pe dataset
python test_model.py --auto

# Test interactiv
python test_model.py
```

### ⚡ Hardware Requirements

**Minim recomandat:**
- GPU: GTX 1060 6GB sau mai bun
- RAM: 8GB
- Spațiu: 10GB liber

**Optimizat pentru:**
- **RTX 4050** (6GB VRAM) - 47 minute training
- **RTX 3060** sau superior - mai rapid
- **CPU only** - posibil dar foarte lent

### 🔧 Troubleshooting

**Probleme comune:**

1. **CUDA nu este detectat**
   ```bash
   # Verifică instalarea
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Out of Memory**
   ```python
   # Reduce batch size în train_mushroom_model.py
   batch_size = 2  # în loc de 4
   ```

3. **Dependințe lipsă**
   ```bash
   python pre_training_check.py  # Va identifica problemele
   ```

### 📧 Support

Pentru probleme cu antrenarea:
1. Verifică `pre_training_check.py`
2. Consultă `TRAINING_RESULTS.md`
3. Verifică log-urile din `runs/detect/`

---

**Nota**: Modelul este optimizat specific pentru RTX 4050 dar funcționează pe orice GPU modern cu minim 4GB VRAM.
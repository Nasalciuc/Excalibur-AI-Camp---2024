<<<<<<< HEAD
# Mushroom Identifier - YOLOv11 Project
## 🍄 AI-Powered Mushroom Detection & Classification

Proiect de detecție și clasificare ciuperci folosind YOLOv11, optimizat pentru RTX 4050.

### 📋 Despre Proiect

Acest sistem de inteligență artificială poate detecta și clasifica 3 tipuri de ciuperci:

- **🟢 Chanterelle (Galbiori)** - Comestibili, siguri
- **🔴 Death-cap (Coprini)** - EXTREM DE TOXICI ⚠️
- **🟢 Field Mushroom (De câmp)** - Comestibili, siguri

### 🎯 Scopul Proiectului

**Siguranța în identificarea ciupercilor** - Prevenirea otrăvirilor accidentale prin identificarea automată a speciilor toxice vs. comestibile.

### 📊 Dataset

- **963 imagini** adnotate în format YOLOv11
- **3 clase** de ciuperci 
- **Augmentare automată** pentru îmbunătățirea performanțelor
- Imagini redimensionate la **640x640px** cu optimizări pentru **480px** pe GPU-uri mici

### 🖥️ Cerințe Hardware

**Recomandat:**
- GPU: RTX 4050+ (6GB VRAM minim)
- RAM: 16GB+
- Stocare: 10GB liberi

**Optimizat pentru RTX 4050:**
- Batch size: 4
- Image size: 480px
- Mixed precision training
- Model: YOLOv11 nano

### 🚀 Instalare și Configurare

1. **Clonează repository-ul:**
```bash
git clone https://github.com/Nasalciuc/Excalibur-AI-Camp---2024.git
cd "mushroom identifier.v2i.yolov11 (1)"
```

2. **Instalează dependințele:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib pillow pyyaml pandas seaborn
```

3. **Verifică GPU-ul:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
```

### 🎓 Training

**Pentru RTX 4050 (6GB VRAM):**
```bash
python train_mushroom_model.py
```

**Parametri optimizați:**
- Epoci: 100
- Batch size: 4
- Learning rate: 0.001
- Mixed precision: Activat
- Early stopping: 15 epoci

**Timp estimat:** 6-8 ore pe RTX 4050

### 🧪 Testare

**Test automat:**
```bash
python test_model.py
```

**Test interactiv:**
```bash
python test_model.py
# Alegeți opțiunea 2 și specificați calea către imagine
```

### 📁 Structura Proiectului

```
mushroom-identifier/
├── data.yaml                    # Configurație dataset
├── train_mushroom_model.py      # Script antrenare optimizat RTX 4050
├── test_model.py               # Script testare și evaluare
├── install_requirements.txt    # Lista dependințelor
├── train/                      # Date de antrenare
│   ├── images/                 # Imagini antrenare
│   └── labels/                 # Etichete YOLO
├── valid/                      # Date de validare
│   ├── images/
│   └── labels/
├── test/                       # Date de testare
│   ├── images/
│   └── labels/
└── runs/                       # Rezultate antrenare (generat automat)
    └── detect/
        └── mushroom_detector_rtx4050/
            └── weights/
                └── best.pt     # Cel mai bun model antrenat
```

### 📊 Rezultate Așteptate

**Metrici de performanță:**
- **mAP50**: >85% (Mean Average Precision la IoU 0.5)
- **mAP50-95**: >70% (Mean Average Precision IoU 0.5-0.95)
- **Precision**: >90% pentru death-cap detection (critic pentru siguranță)
- **Recall**: >85% pentru toate clasele

### ⚠️ Avertismente Importante

**🚨 SIGURANȚA ESTE PRIORITATEA #1:**

1. **Nu vă bazați 100% pe AI** pentru identificarea ciupercilor în natură
2. **Consultați întotdeauna un specialist** pentru ciupercile pe care doriți să le consumați
3. **Death-cap sunt extrem de toxice** - chiar și cantități mici pot fi letale
4. **În caz de îndoială, NU consumați** ciuperca

### 🔬 Despre Dataset

Dataset furnizat de **Roboflow** sub licența **CC BY 4.0**.
- Sursă: [Roboflow Universe](https://universe.roboflow.com/workspace-zbvbx/mushroom-identifier-lyqa4)
- Versiune: 2.0
- Export: YOLOv11 format

### 🏆 Excalibur AI Camp 2024

Acest proiect face parte din **Excalibur AI Camp 2024** - o inițiativă educațională pentru învățarea aplicațiilor practice ale inteligenței artificiale în domenii cu impact real asupra siguranței oamenilor.

**Instructor:** [Numele Instructorului]  
**Participanți:** [Lista Participanților]  
**Durată:** [Perioada Cursului]

### 📞 Contact și Suport

Pentru întrebări despre proiect sau probleme tehnice:
- **GitHub Issues**: [Link către issues]
- **Email**: [Email contact]
- **Discord**: [Server Discord]

### 📄 Licență

Acest proiect este licențiat sub **MIT License**.
Dataset licențiat sub **CC BY 4.0**.

---

**⭐ Dacă acest proiect v-a fost util, vă rugăm să îi dați un star pe GitHub!**

**🤝 Contribuțiile sunt binevenite!** Vezi [CONTRIBUTING.md](CONTRIBUTING.md) pentru detalii.
=======
# Excalibur-AI-Camp---2024
>>>>>>> 84a7de3a58eee27c3b13908b77593c9d75c94eda

# 📷 Radial Distortion Estimation from Pinhole Images

This project implements a pipeline to **detect pinhole centroids in an image and estimate radial distortion** using a least squares approach.

It was developed as part of a **Bachelor's Thesis (TFG)** focused on image processing and optical calibration.

The system includes:

- Image preprocessing (dark / flat correction)
- Pinhole detection
- Subpixel centroid estimation
- Radial distortion estimation
- Simulation tools for validation
- Visualization utilities

---

# 🧠 Method Overview

The pipeline follows these main steps:

## 1. Preprocessing
- Dark frame correction
- Flat-field correction

## 2. Centroid estimation
- Local maxima detection
- Neighbourhood validation
- Window extraction
- Adaptive thresholding per window
- Binary mask extraction
- centroid computed using an **intensity-weighted average**

## 3. Ideal pinholes estimation

## 4. Distortion estimation

---

# 📂 Project Structure

```
Codigo/src/
│
├── main_distortion.py        # Distortion estimation pipeline
├── main_simulation.py        # Synthetic image simulation
│
├── Distortion/
│   ├── DistortionDetector.py
│   └── K_VALUES.py
│
├── Processing/
│   ├── FlatDark.py
│   └── OpticCenter.py
│
├── Simulation/
│   └── ImageSimulator.py
│
├── Utils/
│   └── utils.py
│
├── Visualization/
│   ├── CentroFlat.py
│   ├── CentroidValidation.py
│   ├── ErrorVisual.py
│   └── ErrorVisual2.py
│
└── requirements.txt
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/distortion-estimation.git
cd distortion-estimation
```

Install dependencies:

```bash
pip install -r src/requirements.txt
```

---

# ▶️ Usage

Run the distortion estimation pipeline:

```bash
python Codigo/src/main_distortion.py
```

Run the simulation environment:

```bash
python Codigo/src/main_simulation.py
```

---

# 🔬 Applications

This project can be used for:

- Optical system calibration
- Detector characterization
- Distortion analysis
- Computer vision preprocessing

---

# 👨‍💻 Author

**Xabier Yoldi**

---

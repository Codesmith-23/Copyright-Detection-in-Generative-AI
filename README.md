# 📌 Copyright Detection in Generative AI

A comprehensive deep learning system designed to detect synthetic media manipulation in both **Video** and **Audio**. The system uses a **microservices architecture** with a React frontend and two specialized Flask backends.

---

## 🚀 Features

- 🎥 **Video Detection:** Uses ResNet + LSTM to analyze frame sequences for visual anomalies.  
- 🔊 **Audio Detection:** Analyzes spectral patterns (MFCCs/Mel Spectrograms) to detect AI-generated voice.  
- 🔗 **Multimodal Fusion:** Combines video and audio analysis with confidence blending.  
- 🔐 **JWT Authentication:** Secure user registration and login with token-based session management.  
- 📊 **Analysis History:** Track, view, and delete past analysis results.  
- 🧑‍💻 **Creator Registration:** Protect identity against unauthorized deepfakes.  
- 🛡 **Copyright Detection:** Matches flagged faces with registered protected identities.  
- 🧠 **Identity Matching:** Uses FaceNet embeddings for high-accuracy face comparison (0.6+ threshold).  
- 📜 **License Status Tracking:** Identifies unauthorized usage of protected identities.  
- ☁️ **Automated Model Fetching:** Downloads latest trained models from **Hugging Face** on first startup.  
- 📑 **Detailed Reporting:** Generates reports with confidence scores, flagged frames, and audio segments.  
- 🔒 **Zero Retention Policy:** Uploaded files are deleted immediately after processing.  
- ⚙️ **Service Orchestration:** Launch all 4 microservices with a single command.  

---

## 🧰 Tech Stack

- **Frontend:** React.js / Next.js (TypeScript)  
- **Video Backend:** Python, Flask, TensorFlow/Keras, OpenCV, Dlib  
- **Audio Backend:** Python, Flask, PyTorch, Librosa  
- **Identity Service:** Python, Flask, FaceNet, OpenCV DNN  
- **Database:** SQLite3 (User accounts, analysis history, protected identities, license records)  
- **Model Hosting:** Hugging Face Hub  
- **Authorization:** JWT (JSON Web Tokens)  
- **Orchestration:** Multi-service launcher (`orchestrator.py`)  

---

## ⚙️ Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.11  
- Node.js & npm  
- Git  
- **FFmpeg** (Required for audio/video processing)  

```bash
# Windows
winget install ffmpeg

# Mac
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

---

## 🛠 Installation & Setup

### 🔽 Clone the Repository
```bash
git clone https://github.com/Codesmith-23/Deepfake-Detection-System.git
cd Deepfake-Detection-System
```

---

## 🎧 Setting up Audio Detection

### Open a new terminal
```bash
cd "Voice Deepfake"
```

### Create virtual environment (optional)
```bash
python -m venv venv
```

#### Activate environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Audio Service
```bash
python api_audio.py
```

---

## 🎥 Setting up Video Detection

### Open a new terminal
```bash
cd "DeepFake/backend"
```

### Create virtual environment
```bash
python -m venv venv
```

#### Activate environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Video Service
```bash
python app.py
```

---

## 🌐 Frontend Setup

### Open a new terminal
```bash
cd "DeepFake/frontend"
```

### Install dependencies
```bash
npm install
```

### Start the app
```bash
npm run dev
```

---

## ⚡ Quick Start with Orchestrator

Run all services using a single command:

```bash
python orchestrator.py
```

### This will start:
- Identity Service → Port 5002  
- Audio API → Port 5001  
- Video Backend → Port 5000  
- Frontend → Port 3000  

> ⚠️ **Note:** Configure the Python paths in `orchestrator.py` before first use.

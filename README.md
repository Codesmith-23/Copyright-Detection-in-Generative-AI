# 📌 Copyright Detection in Generative AI

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

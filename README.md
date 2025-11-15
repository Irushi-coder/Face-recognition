# Face Recognition Attendance System

A real-time face recognition system for automated attendance marking using YOLOv8 and deep learning-based face embeddings.

## 🎯 Features
- Face recognition with deep learning embeddings
- Automated attendance logging
- Cosine similarity-based matching
- Support for multiple faces per frame

## 📋 Prerequisites

- Python 2.8+
- CUDA-compatible GPU (recommended)

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/Irushi-coder/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Quick Start
```bash
python scripts/run_attendance.py
```

### Extract Face Features
```bash
python scripts/extract_features.py --input data/faces --output data/vectors
```

## 🤝 Contributing

This is a final year project contribution. For questions or suggestions, please open an issue.

## 👤 Author

**Your Name**
- GitHub: [@Irushi-coder](https://github.com/Irushi-coder)
- University: University of Peradeniya
- Project Year: 2024/2025

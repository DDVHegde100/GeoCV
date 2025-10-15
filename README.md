# GeoCV - Real-Time Computer Vision GeoGuesser AI

**An AI-powered system that plays GeoGuesser using computer vision and OpenCV to analyze street view images and compete against human players in real-time.**

## 🎯 Project Overview

GeoCV combines computer vision, geospatial reasoning, and web technologies to create an interactive GeoGuesser AI that analyzes visual cues from street view images to predict geographic locations. The system emphasizes OpenCV-based computer vision techniques while providing a challenging but fair competition against human players.

## 🏗️ Architecture

```
Backend (Python FastAPI + OpenCV)
├── Computer Vision Pipeline
│   ├── Object Detection (Cars, Signs, Architecture)
│   ├── Semantic Segmentation (Roads, Vegetation, Terrain)
│   ├── OCR (Street Signs, Language Detection)
│   └── Environmental Analysis (Sky, Shadows, Sun Position)
├── Geospatial Reasoning Engine
├── Machine Learning Models
└── API Endpoints

Frontend (React/Next.js)
├── Game Interface
├── AI Visualization Panel
├── Human Player Panel
└── Results & Scoring System
```

## 🔬 Computer Vision Features

### Primary OpenCV-Based Analysis
- **Vehicle Detection & Classification**: Regional car models and styles
- **Road & Infrastructure Analysis**: Lane markings, road materials, signage
- **Vegetation & Terrain Recognition**: Trees, climate zones, urban density
- **Architecture Analysis**: Building styles, regional construction patterns
- **Environmental Cues**: Sky conditions, shadow analysis for sun position
- **Text & Language Detection**: OCR on signs, street names, shop fronts

### Feature Extraction Pipeline
1. **Preprocessing**: Image enhancement, noise reduction
2. **Multi-scale Detection**: Objects at various scales and orientations
3. **Feature Aggregation**: Combining multiple visual cues
4. **Confidence Scoring**: Reliability assessment of each detected feature
5. **Geospatial Mapping**: Feature-to-location probability mapping

## 🛠️ Tech Stack

### Backend
- **Python 3.9+**
- **OpenCV 4.x** (Primary CV library)
- **FastAPI** (Web framework)
- **NumPy** (Numerical computing)
- **Scikit-learn** (ML models)
- **Tesseract/EasyOCR** (Text recognition)
- **Pillow** (Image processing)
- **Requests** (API calls)

### Frontend
- **React/Next.js**
- **TypeScript**
- **Tailwind CSS**
- **Leaflet/MapBox** (Interactive maps)
- **Socket.IO** (Real-time communication)

### Development & Deployment
- **Docker** (Containerization)
- **GitHub Actions** (CI/CD)
- **Vercel** (Frontend deployment)
- **Railway/Render** (Backend deployment)

## 📁 Project Structure

```
geocv/
├── backend/
│   ├── app/
│   │   ├── core/           # Core CV pipeline
│   │   ├── models/         # ML models and data structures
│   │   ├── api/           # FastAPI routes
│   │   ├── services/      # Business logic
│   │   └── utils/         # Helper functions
│   ├── data/              # Training data and models
│   ├── tests/             # Backend tests
│   └── requirements.txt
├── frontend/
│   ├── components/        # React components
│   ├── pages/            # Next.js pages
│   ├── hooks/            # Custom hooks
│   ├── utils/            # Frontend utilities
│   └── styles/           # CSS and styling
├── docs/                 # Documentation
├── scripts/              # Utility scripts
└── docker-compose.yml    # Development environment
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (optional)
- Google Maps API key (for Street View images)

### Installation

1. **Clone and setup backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Setup frontend:**
```bash
cd frontend
npm install
```

3. **Environment configuration:**
```bash
cp .env.example .env
# Add your API keys and configuration
```

### Development

**Start backend:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Start frontend:**
```bash
cd frontend
npm run dev
```

## 🎮 Game Modes

1. **Classic Mode**: Unlimited time for both AI and human
2. **Speed Mode**: AI gets fixed time, human gets AI time + 10 seconds
3. **Training Mode**: See AI's reasoning process step-by-step
4. **Challenge Mode**: Multiple rounds with scoring

## 🔬 CV Pipeline Details

### Detection Models
- **Object Detection**: Custom OpenCV cascades + traditional CV methods
- **Semantic Segmentation**: Color-based and texture analysis
- **OCR Integration**: Tesseract with OpenCV preprocessing
- **Feature Matching**: Template matching for common regional elements

### Geospatial Reasoning
- **Regional Databases**: Car models, architecture, vegetation by region
- **Climate Mapping**: Weather patterns and seasonal indicators
- **Infrastructure Patterns**: Road styles, signage systems by country
- **Cultural Markers**: Architectural styles, urban planning patterns

## 📊 Evaluation Metrics

- **Distance Error**: Kilometers from actual location
- **Country Accuracy**: Correct country prediction rate
- **Feature Confidence**: Reliability of individual CV detections
- **Response Time**: Speed of AI analysis and prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Links

- [Project Documentation](./docs/)
- [API Documentation](./docs/api.md)
- [CV Pipeline Details](./docs/cv-pipeline.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

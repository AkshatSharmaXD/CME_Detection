# Backend - CME Detection & Space Weather Monitoring API

**Made for Team Digi Shakti - Smart India Hackathon (SIH)**

This is the FastAPI backend server that powers the CME Detection and Space Weather Monitoring System. It provides RESTful API endpoints for real-time space weather data, CME detection, forecasting, and satellite data analysis.

## 🎯 Overview

The backend is built using FastAPI and provides comprehensive APIs for:
- Real-time space weather data fetching from NOAA
- CME (Coronal Mass Ejection) detection using machine learning algorithms
- 7-day space weather forecasting using LSTM models
- Geomagnetic storm monitoring and prediction
- Satellite field data matching and CME probability calculation
- Historical data analysis and visualization

## 🛠️ Tech Stack

- **FastAPI 0.104.1**: Modern, fast web framework for building APIs
- **Python 3.11+**: Core programming language
- **Uvicorn**: ASGI server for running FastAPI
- **Pandas & NumPy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning algorithms for CME detection
- **SQLAlchemy**: Database ORM (PostgreSQL support)
- **Requests & aiohttp**: HTTP clients for external API calls
- **cdflib**: CDF file parsing for space weather data
- **BeautifulSoup4**: Web scraping for NOAA data
- **Matplotlib & Plotly**: Data visualization and chart generation

## 📋 Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- PostgreSQL (optional, for database features)
- Virtual environment (recommended)

## 🚀 Installation

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment (Optional)

Create a `.env` file in the backend directory for database configuration:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/cme_detection
```

## 🏃 Running the Server

### Development Mode

```bash
python main.py
```

The server will start on `http://localhost:8002`

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8002 --workers 4
```

## 📁 Project Structure

```
backend/
├── main.py                      # Main FastAPI application
├── noaa_realtime_data.py        # NOAA Space Weather data fetcher
├── database.py                  # Database models and configuration
├── db_service.py               # Database service layer
├── omniweb_data_fetcher.py     # NASA OMNIWeb data fetcher
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
│
├── api/                        # API modules
│   └── model_accuracy.py       # Model accuracy endpoints
│
├── scripts/                    # Core detection and ML scripts
│   ├── comprehensive_cme_detector.py    # Main CME detection algorithm
│   ├── halo_cme_detector.py            # Halo CME detection
│   ├── real_data_sync.py               # Real-time data synchronization
│   ├── data_validator.py               # Data validation utilities
│   ├── swis_data_loader.py              # SWIS data loader
│   ├── cactus_scraper.py                # CACTUS CME scraper
│   └── 7_days_forecast/                # Forecast model scripts
│       ├── train_model.py              # LSTM model training
│       ├── forecast_predictor.py       # Forecast prediction
│       └── model files (.npy, .pkl)    # Trained model files
│
├── data/                       # Data storage
│   └── aditya_l1/             # Aditya L1 mission data
│
├── downloads/                  # Downloaded data files
├── output/                     # Generated outputs (CSV, etc.)
└── venv/                       # Virtual environment (gitignored)
```

## 🔌 API Endpoints

### Data Endpoints

#### `GET /api/data/summary`
Get comprehensive data summary and system status.

**Response:**
```json
{
  "success": true,
  "total_records": 254228,
  "date_range": {
    "start": "1995-01-01",
    "end": "2024-12-31"
  },
  "parameters": ["velocity", "density", "temperature", "bz"],
  "system_status": "operational"
}
```

#### `GET /api/data/realtime`
Get real-time solar wind data from NOAA.

**Response:**
```json
{
  "success": true,
  "data_source": "NOAA Combined",
  "timestamp": "2025-01-XX...",
  "solar_wind": {
    "speed": 450.0,
    "density": 5.2,
    "temperature": 105000,
    "bz_gsm": -3.2
  }
}
```

#### `GET /api/data/particle`
Get particle data for visualization.

#### `POST /api/data/upload`
Upload CDF file for analysis.

**Request:**
- `file`: CDF file (multipart/form-data)
- `run_detection`: boolean (optional)

### CME Detection Endpoints

#### `GET /api/cme/recent`
Get recent CME events with detection details.

**Query Parameters:**
- `days`: Number of days to look back (default: 14)
- `limit`: Maximum number of events (default: 50)

#### `POST /api/ml/analyze-cdf`
Analyze uploaded CDF file for CME detection.

**Request:**
- `file`: CDF file

**Response:**
```json
{
  "success": true,
  "detection_results": {
    "total_events": 3,
    "all_events": [...],
    "analysis_summary": "..."
  }
}
```

### Geomagnetic Data Endpoints

#### `GET /api/geomagnetic/storm/live`
Get live geomagnetic indices (Kp, DST, Ap, F10.7).

#### `GET /api/forecast/predictions`
Get 7-day space weather forecast predictions.

**Response:**
```json
{
  "success": true,
  "forecast": {
    "parameters": ["Dst", "Kp", "Ap", "Sunspot"],
    "timestamps": [...],
    "predictions": {...}
  },
  "model_accuracy": 97.3
}
```

### Satellite Data Endpoints

#### `GET /api/satellites`
Get list of available satellites from external API.

**Response:**
```json
{
  "success": true,
  "satellites": [
    {
      "norad_id": 25544,
      "name": "ISS",
      "object_type": "PAYLOAD"
    }
  ]
}
```

#### `GET /api/satellites/{norad_id}`
Get detailed information for a specific satellite.

#### `GET /api/satellites/{norad_id}/cme-prediction`
Get CME probability for satellite based on coordinate matching with NOAA wind data.

**Query Parameters:**
- `threshold`: Probability threshold (default: 0.5)

**Response:**
```json
{
  "success": true,
  "satellite": {...},
  "noaa_match": {
    "distance": 2.3,
    "wind_data": {...}
  },
  "cme_probability": 0.75,
  "risk_level": "moderate",
  "scores": {
    "speed": 0.8,
    "density": 0.7,
    "temperature": 0.6,
    "bz": 0.9
  }
}
```

### NOAA Integration Endpoints

#### `GET /api/noaa/alerts`
Get space weather alerts from NOAA.

#### `GET /api/noaa/solar-flares`
Get solar flare data.

#### `GET /api/noaa/images/{source}`
Get image sequences for animations.

**Path Parameters:**
- `source`: `lasco-c3`, `lasco-c2`, `suvi-094`, `enlil`, `ovation-north`, `ovation-south`

**Query Parameters:**
- `count`: Number of images (default: 1)

## 🧠 Core Algorithms

### CME Detection Algorithm

The system uses a multi-parameter approach:

1. **Velocity Threshold Detection**: Identifies high-speed solar wind streams (>600 km/s)
2. **Density Spike Detection**: Detects sudden density increases
3. **Temperature Anomaly Detection**: Monitors proton temperature variations
4. **Bz Component Analysis**: Tracks southward magnetic field (Bz < -10 nT)
5. **Composite Scoring**: Weighted combination of all parameters

**Location**: `scripts/comprehensive_cme_detector.py`

### Forecast Model

- **Model Type**: LSTM (Long Short-Term Memory) neural network
- **Training Data**: 29 years of historical space weather data (1995-2024)
- **Parameters**: DST Index, Kp Index, Ap Index, Sunspot Number
- **Accuracy**: 97.3% overall accuracy
- **Forecast Period**: 7 days ahead
- **Location**: `scripts/7_days_forecast/`

### Composite Index Calculation

Uses Principal Component Analysis (PCA) to combine:
- DST Index (Disturbance Storm Time)
- Kp Index (Planetary K-index)
- Ap Index (Daily geomagnetic activity)
- Sunspot Number

**Location**: `main.py` (lines 2895-3135)

## 🔧 Configuration

### config.yaml

Edit `config.yaml` to configure:
- Database connection settings
- NOAA API endpoints
- Model parameters
- Detection thresholds
- Data source preferences

### Environment Variables

Create `.env` file for sensitive configuration:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/cme_detection
NOAA_API_KEY=your_api_key_here
```

## 📊 Data Sources

### NOAA Space Weather Prediction Center
- **Base URL**: `https://services.swpc.noaa.gov/`
- **Endpoints**:
  - Solar Wind: `/products/solar-wind/`
  - Geomagnetic: `/products/geospace/`
  - Images: `/products/animations/`
  - Alerts: `/products/alerts/`

### NASA OMNIWeb
- Historical space weather data
- Multi-spacecraft merged data

### External APIs
- Satellite API: `https://sat-api-k1ga.onrender.com/api/satellites/`
- Aditya L1 mission data (when available)

## 🧪 Testing

### Test Individual Endpoints

```bash
# Test data summary
curl http://localhost:8002/api/data/summary

# Test realtime data
curl http://localhost:8002/api/data/realtime

# Test CME detection
curl http://localhost:8002/api/cme/recent?days=7
```

### Run Test Scripts

```bash
# Test NOAA data fetching
python test_all_params.py

# Test CDF upload
python test_cdf_upload.py

# Test data validation
python scripts/test_data_validation.py
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Windows: Find and kill process
netstat -ano | findstr :8002
taskkill /PID <PID> /F

# Linux/Mac: Find and kill process
lsof -ti:8002 | xargs kill -9
```

### Database Connection Errors

- Ensure PostgreSQL is running
- Check `DATABASE_URL` in `.env` or `config.yaml`
- Verify database credentials

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### CDF File Parsing Errors

- Ensure `cdflib` is installed: `pip install cdflib`
- Check CDF file format and version
- Verify file contains required variables

### NOAA API Errors

- Check internet connection
- Verify NOAA endpoints are accessible
- Some endpoints may have rate limits

## 📈 Performance Optimization

- **Caching**: Implement Redis for frequently accessed data
- **Database Indexing**: Add indexes on frequently queried columns
- **Async Operations**: Use async/await for I/O-bound operations
- **Connection Pooling**: Configure SQLAlchemy connection pool
- **Rate Limiting**: Implement rate limiting for external API calls

## 🔒 Security Considerations

- **CORS**: Configured for frontend origin only
- **Input Validation**: All inputs validated using Pydantic models
- **File Upload**: Size limits and type validation for CDF files
- **Error Handling**: Generic error messages to prevent information leakage

## 📝 Logging

Logs are written to console with different levels:
- **INFO**: General information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors
- **DEBUG**: Detailed debugging information

Configure logging level in `main.py`:
```python
logging.basicConfig(level=logging.INFO)
```

## 🚀 Deployment

### Docker Deployment

```bash
docker build -t cme-backend .
docker run -p 8002:8002 cme-backend
```

### Production Server

Use Gunicorn with Uvicorn workers:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8002
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NOAA Space Weather APIs](https://www.swpc.noaa.gov/)
- [NASA OMNIWeb](https://omniweb.gsfc.nasa.gov/)

---

## 👥 Development Team

**Made for Team Digi Shakti - Smart India Hackathon (SIH)**

**End to end developed and produced by me as Tech System Lead & Full Stack Architect**

### 🙏 Special Thanks to Team Members:

- **Akshat Sharma**
- **Mayank Saini**
- **Deepak Singh**
- **Garima**
- **Lily**

---

**Note**: This backend is part of the CME Detection & Space Weather Monitoring System. For frontend documentation, see `../frontend/README.md`.


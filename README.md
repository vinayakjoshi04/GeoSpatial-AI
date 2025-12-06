# 🌤️ GeoSpatial AI - India Weather & Pollution Advisor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geospatial-ai-bse6n3xbxgwwktd9xmkyph.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AI-Powered Weather & Air Quality Predictions for Indian Cities**

A comprehensive machine learning application that predicts weather conditions and air quality metrics for cities across India, powered by advanced ML models and Google's Gemini AI for personalized health recommendations.

## 🌟 Features

### 🎯 Core Capabilities
- **🌡️ Weather Predictions**: Temperature, humidity, wind speed, and atmospheric pressure forecasts
- **🌫️ Air Quality Monitoring**: Real-time AQI predictions with detailed pollutant breakdowns (PM2.5, PM10, NO2, SO2, CO, O3)
- **🤖 AI-Powered Advice**: Personalized health recommendations using Google Gemini 2.0 Flash
- **🗺️ Interactive Mapping**: Visualize and explore Indian cities with geospatial coordinates
- **📊 Multi-City Comparison**: Compare weather and pollution metrics across multiple cities
- **📈 Data Visualization**: Interactive charts and gauges for intuitive data interpretation

### 💡 Smart Recommendations
- Activity-specific guidance (running, cycling, outdoor activities)
- Health warnings based on environmental conditions
- Vulnerable group alerts (children, elderly, respiratory conditions)
- Mask and indoor/outdoor activity suggestions

## 🚀 Live Demo

**Access the application here:** [GeoSpatial AI Web App](https://geospatial-ai-bse6n3xbxgwwktd9xmkyph.streamlit.app/)

## 📸 Screenshots

### Home Dashboard
Clean and intuitive interface with quick access to all features.

### Interactive City Map
Explore Indian cities with geospatial visualization and select locations for predictions.

### Weather & Pollution Predictions
Comprehensive forecasts with AI-powered health recommendations.

### Multi-City Comparison
Side-by-side analysis of weather and air quality across cities.

## 🛠️ Technology Stack

### Machine Learning
- **Scikit-learn**: Model training and preprocessing
- **XGBoost**: Gradient boosting for improved predictions
- **RandomForest**: Ensemble learning algorithms
- **Neural Networks (MLP)**: Deep learning for complex patterns

### AI & NLP
- **Google Gemini 2.0 Flash**: Personalized health advice generation
- **Custom Prompt Engineering**: Context-aware recommendations

### Data Processing & Visualization
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Plotly**: Interactive charts and visualizations
- **Joblib**: Model serialization and loading

### Web Framework
- **Streamlit**: Responsive web application interface

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API Key ([Get it here](https://aistudio.google.com/app/apikey))
- Git (for cloning the repository)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vinayakjoshi04/GeoSpatial-AI.git
cd GeoSpatial-AI
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Prepare Data and Models

#### Option A: Use Pre-trained Models
If you have pre-trained models in the `models/` directory, skip to step 6.

#### Option B: Train Models from Scratch
```bash
# Train weather prediction model
python India_Weather.py

# Train pollution prediction model
python India_Pollution.py
```

This will generate:
- `models/weather_best_model.pkl`
- `models/weather_scaler.pkl`
- `models/pollution_best_model.pkl`
- `models/pollution_scaler.pkl`

### 6. Verify Gemini API Setup
```bash
python test_gemini_models.py
```

## 🚀 Running the Application

### Local Development
```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Production Deployment
The app is deployed on Streamlit Cloud. For your own deployment:

1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add `GEMINI_API_KEY` to Secrets in Streamlit Cloud settings
5. Deploy!

## 📁 Project Structure

```
GeoSpatial-AI/
│
├── data/
│   ├── india_weather.csv          # Historical weather data
│   ├── india_pollution.csv        # Historical pollution data
│   └── india_cities_latlon.csv    # City coordinates
│
├── models/
│   ├── weather_best_model.pkl     # Trained weather model
│   ├── weather_scaler.pkl         # Weather data scaler
│   ├── pollution_best_model.pkl   # Trained pollution model
│   └── pollution_scaler.pkl       # Pollution data scaler
│
├── gemini_advice.py               # AI advice generation
├── India_Weather.py               # Weather model training
├── India_Pollution.py             # Pollution model training
├── streamlit_app.py               # Main web application
├── test_gemini_models.py          # API testing utility
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (not in repo)
└── README.md                      # This file
```

## 📊 Data Sources

### Datasets Required
1. **india_weather.csv**: Historical weather data with columns:
   - `city`, `date`, `meantemp`, `humidity`, `wind_speed`, `meanpressure`

2. **india_pollution.csv**: Historical pollution data with columns:
   - `city`, `date`, `pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `aqi_proxy`

3. **india_cities_latlon.csv**: City coordinates with columns:
   - `city`, `latitude`, `longitude`

### Data Format
- All dates should be in `YYYY-MM-DD` format
- Numeric values for all measurements
- Consistent city names across all datasets

## 🔍 Model Performance

The application uses ensemble learning with three models:
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Multi-Layer Perceptron (MLP)**

The best-performing model is automatically selected based on Mean Squared Error (MSE) during training.

### Model Evaluation Metrics
- **Weather Model MSE**: Evaluated on temperature, humidity, wind speed, and pressure
- **Pollution Model MSE**: Evaluated on PM2.5, PM10, NO2, SO2, CO, O3, and AQI

## 🤖 AI Advice System

### How It Works
1. **Context Analysis**: Evaluates current weather and pollution conditions
2. **Risk Assessment**: Identifies primary concerns (temperature, AQI, specific pollutants)
3. **Personalized Recommendations**: Generates activity-specific advice using Gemini AI
4. **Fallback System**: Rule-based advice when API is unavailable

### Sample Prompts
- "Should I go for a run today?"
- "Is it safe for my kids to play outside?"
- "Can I go cycling this evening?"
- "Should I wear a mask?"

## 🎨 User Interface

### Navigation Pages
1. **🏠 Home**: Overview and quick statistics
2. **📍 City Selection & Map**: Interactive map for city selection
3. **🔮 Predictions & Advice**: Weather/pollution forecasts with AI recommendations
4. **📊 City Comparison**: Multi-city analysis and comparison

### Color-Coded AQI System
- 🟢 **Good (0-50)**: Safe for all activities
- 🟡 **Moderate (51-100)**: Acceptable for most
- 🟠 **Unhealthy for Sensitive Groups (101-150)**: Caution advised
- 🔴 **Unhealthy (151-200)**: Everyone may experience effects
- 🟣 **Very Unhealthy (201-300)**: Health alert
- 🟤 **Hazardous (301+)**: Emergency conditions

## 🔐 API Configuration

### Gemini API Setup
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create or select a project
3. Generate an API key
4. Add to `.env` file: `GEMINI_API_KEY=your_key_here`

### Rate Limits
- Free tier: 60 requests per minute
- Fallback system activates if API limits exceeded

## 🐛 Troubleshooting

### Common Issues

**Issue**: Models not loading
```bash
# Solution: Retrain models
python India_Weather.py
python India_Pollution.py
```

**Issue**: Gemini API errors
```bash
# Solution: Test API configuration
python test_gemini_models.py
```

**Issue**: Missing dependencies
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --upgrade
```

**Issue**: Data loading errors
- Ensure all CSV files are in the `data/` directory
- Check file formats match expected columns
- Verify no corrupted data entries

## 📈 Future Enhancements

- [ ] Real-time data integration from weather APIs
- [ ] Historical trend analysis and forecasting
- [ ] Mobile application development
- [ ] User accounts and saved preferences
- [ ] Email/SMS alert system
- [ ] Integration with more Indian cities
- [ ] Multi-language support
- [ ] Advanced data visualizations

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Test thoroughly before submitting
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vinayak Joshi**

- GitHub: [@vinayakjoshi04](https://github.com/vinayakjoshi04)
- Project Link: [GeoSpatial-AI](https://github.com/vinayakjoshi04/GeoSpatial-AI)

## 🙏 Acknowledgments

- **Google Gemini AI** for powering personalized health recommendations
- **Streamlit** for the excellent web framework
- **Scikit-learn, XGBoost** for robust ML capabilities
- **Plotly** for interactive visualizations
- Indian meteorological and pollution monitoring agencies for data insights

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/vinayakjoshi04/GeoSpatial-AI/issues)
- Check existing issues before creating new ones
- Provide detailed information for bug reports

## ⭐ Star the Repository

If you find this project helpful, please consider giving it a star on GitHub!

---

<div align="center">

**Made with ❤️ for cleaner air and better health decisions in India**

[🌐 Live Demo](https://geospatial-ai-bse6n3xbxgwwktd9xmkyph.streamlit.app/) • [📝 Report Bug](https://github.com/vinayakjoshi04/GeoSpatial-AI/issues) • [✨ Request Feature](https://github.com/vinayakjoshi04/GeoSpatial-AI/issues)

</div>
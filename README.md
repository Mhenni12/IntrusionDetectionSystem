# AI-Powered Multi-Class Intrusion Detection System (NSL-KDD)

## Overview

A professional, real-time intrusion detection system dashboard built with Streamlit that simulates network traffic analysis using machine learning models. The system provides explainable AI-based threat detection with a cybersecurity-themed SOC (Security Operations Center) interface.

## Features

### 🎯 Core Capabilities
- **Real-time Detection**: Simulates streaming network flow analysis with live predictions
- **Multi-Class Classification**: Detects 5 types of network activity (Normal, DoS, Probe, R2L, U2R)
- **Multiple ML Models**: Supports SVM, Logistic Regression, Random Forest, and XGBoost
- **Explainability**: Feature importance visualization and prediction explanations
- **Interactive Visualizations**: Real-time charts, confusion matrices, and 2D projections

### 🛡️ Dashboard Components

1. **Control Panel (Sidebar)**
   - Model selection dropdown
   - Start/Stop simulation controls
   - Speed adjustment (1x to 10x)
   - Live statistics counters

2. **Real-Time Detection Panel**
   - Current flow details (ID, timestamp, IPs)
   - Predicted class with confidence score
   - Color-coded severity indicators
   - Alert banners for detected threats
   - Key feature values display

3. **Attack Distribution Overview**
   - Pie chart showing class distribution
   - Bar chart with attack counts
   - Time-series plot of cumulative attacks

4. **Model Performance Panel**
   - Overall metrics (Accuracy, F1-scores)
   - Per-class recall table
   - Confusion matrix heatmap

5. **Explainability Panel**
   - Top 10 feature importance visualization
   - Textual explanation of current prediction
   - Model-specific insights

6. **2D Threat Landscape**
   - PCA projection of network flows
   - Color-coded by attack class
   - Highlighted current flow marker

## Installation

### Prerequisites
```bash
Python 3.8+
```

### Required Packages
```bash
pip install streamlit pandas numpy plotly scikit-learn joblib xgboost
```

### Quick Install
```bash
# Clone or download the repository
cd intrusion-detection-system

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run ids_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Model Files

The application expects pre-trained model files in the same directory:
- `svm_model.joblib`
- `logistic_model.joblib`
- `rf_model.joblib`
- `xgb_model.joblib`

**Note**: For demonstration purposes, if model files are not found, the application will create dummy models automatically.

### Using Your Own Models

To use your own trained models:

1. Train your models on the NSL-KDD dataset
2. Save them using joblib:
   ```python
   import joblib
   joblib.dump(your_model, 'model_name.joblib')
   ```
3. Place the model files in the application directory
4. Update the `model_files` dictionary in the code if using different filenames

### Using Your Own Dataset

To use a different dataset:

1. Prepare your data with features and labels
2. Modify the `load_dataset()` function to load your data
3. Ensure features are preprocessed (normalized/encoded)
4. Update class names and feature names accordingly

## Dataset Information

### NSL-KDD Dataset
The NSL-KDD dataset is an improved version of the KDD'99 dataset, commonly used for network intrusion detection research.

**Attack Classes:**
- **Normal**: Legitimate network traffic
- **DoS** (Denial of Service): Attacks that make a machine or network resource unavailable
- **Probe**: Surveillance and reconnaissance attacks
- **R2L** (Remote to Local): Unauthorized access from a remote machine
- **U2R** (User to Root): Unauthorized access to superuser privileges

### Features
The dataset includes 41 features representing various aspects of network connections:
- Duration, protocol type, service, flag
- Source/destination bytes
- Connection counts and rates
- Error rates
- Host-based features

## Customization

### Theme Customization
Modify the CSS in the `st.markdown()` section at the top of the file to customize:
- Colors and gradients
- Fonts (currently using Orbitron, Space Mono, Rajdhani)
- Component styling
- Animations and effects

### Adding New Models
To add support for additional models:

1. Add the model name to the `model_files` dictionary
2. Ensure the model supports `predict()` and `predict_proba()` methods
3. For explainability features, add logic in the explainability section

### Speed Adjustment
Modify the `speed_map` dictionary to change simulation speeds:
```python
speed_map = {'1x': 1.0, '2x': 0.5, '5x': 0.2, '10x': 0.1}
```

## Architecture

### Key Components

- **Session State Management**: Uses `st.session_state` to maintain simulation state
- **Caching**: Models and data are cached using `@st.cache_resource` and `@st.cache_data`
- **Real-time Updates**: Automatic page refresh with `st.rerun()` during simulation
- **Plotly Integration**: Interactive, professional-grade visualizations

### Performance Optimization

- Model loading is cached to prevent reloading on every interaction
- PCA computation is cached for efficient 2D visualization
- Session state prevents unnecessary data resets

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from your repository

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "ids_dashboard.py"]
```

### Local Production
```bash
streamlit run ids_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

## Security Considerations

- This is a demonstration/educational tool
- Does not perform actual network traffic capture
- Model predictions should be validated in production environments
- Implement proper authentication for production deployments

## Troubleshooting

### Models Not Loading
- Ensure model files are in the correct directory
- Check that models were saved with compatible scikit-learn versions
- Verify model file names match the `model_files` dictionary

### Performance Issues
- Reduce simulation speed if experiencing lag
- Decrease dataset size for testing
- Ensure sufficient RAM for model loading

### Visualization Problems
- Update Plotly to the latest version
- Clear browser cache
- Check console for JavaScript errors

## Future Enhancements

- [ ] Integration with live network traffic capture
- [ ] SHAP value integration for deeper explainability
- [ ] Model comparison features
- [ ] Export functionality for reports
- [ ] Anomaly detection algorithms
- [ ] Alert notification system
- [ ] Historical analysis dashboard
- [ ] Model retraining interface

## License

This project is provided as-is for educational and demonstration purposes.

## Credits

- **Dataset**: NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases)
- **Visualization**: Plotly
- **Framework**: Streamlit
- **Design**: Custom cybersecurity-themed UI

## Contact & Support

For questions, issues, or contributions, please open an issue on the project repository.

---

**⚡ Built with Streamlit | 🛡️ Securing Networks with AI**

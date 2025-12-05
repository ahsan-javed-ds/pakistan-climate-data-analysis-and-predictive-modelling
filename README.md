# Pakistan Climate Prediction System: Advanced Machine Learning for Flood Risk Analysis

Pakistan faces severe flooding challenges during July-August monsoon periods, with recent events highlighting the critical need for predictive climate modeling. This project develops a production-ready machine learning system analyzing 116 years (1901-2016) of Pakistan's national climate data to predict rainfall and temperature patterns. The system employs a hybrid modeling architecture with advanced feature engineering for temperature prediction and an optimized simple approach for rainfall forecasting.

**Key Achievement**: The system achieves R² = 0.992 for temperature prediction and R² = 0.716 for rainfall prediction using Pakistan's average climate data, providing reliable forecasting capabilities for flood risk assessment during critical monsoon periods.

## -- Project Context & Significance --

### Pakistan's Flood Challenge
Pakistan experiences devastating floods during July-August monsoon seasons, causing significant economic and humanitarian impacts. This predictive system addresses the urgent need for accurate climate forecasting to support:
- Early warning systems for flood-prone regions
- Agricultural planning and crop management
- Water resource management
- Emergency preparedness protocols

![CNN_Pakistan_flood_image](https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2224810741.jpg?c=16x9&q=h_653,w_1160,c_fill/f_avif)
*Picture Reference: CNN World Article (https://edition.cnn.com/2025/07/17/asia/pakistan-flood-deaths-climate-intl-hnk)*

### Data Coverage
The analysis utilizes comprehensive national average climate data for Pakistan, providing country-level insights while acknowledging regional variations. The 116-year historical dataset enables robust pattern recognition and long-term trend analysis essential for understanding Pakistan's complex monsoon-driven climate system.

---

## -- Production Deployment --

## 1. FastAPI Deployment with MLOps Pipeline

A stateless API with strict data validation and automated CI/CD workflows

### Architecture & Key Feature:

1. `High-Performance API`: Built with **FastAPI** for asynchronous inference and auto-generated Swagger documentation.
2. `Dockerized Deployment`: Fully **containerized** environment ensuring high reproducibility across local dev and prod stages.
3. `CI/CD Automation`: **GitHub Actions** pipeline performs syntax checks and dependency installation on every push.
4. `Effective Error Handling`: Implements a **Fall-back strategy** to work if the models fails or input data lacks the context.
5. `Type Safety`: Utilizes **pydantic** models to validate the inputs.

### Startup Steps:

**Option #1: Run with Docker (Recommended)**

```
# Build the docker image
docker build -t pakistan-climate-engine .

# Run the container
docker run -p 8000:8000 pakistan-climate-engine
```

**Option #2: Run with Python (Locally)**
```
# Install the dependencies
pip install -r requirements.txt

# Start the server
uvicorn fastapi_app:app --reload
```

#### API Documentations:
Once running, access the Swagger Ui at `http://localhost8000/docs`

**Endpoint: `/predict` (POST)**

**Request:**
```
{
   "year": 2025,
   "month": 7
}
```
**Response:**
```
{
  "year": 2025,
  "month": "July",
  "rainfall": 56.6,
  "temperature": 28.9,
  "season": "Summer",
  "is_monsoon": true,
  "success": true,
  "method": "pipeline"
}
```
- `method` field indicates if the prediction came from the ML Pipeline (`pipeline`), the Smart Inference function (`smart_inference`), or the Heuristic Fallback (`heuristic_fallback`)

---

## 2. Streamlit Application
Interactive web application providing:
- Real-time climate predictions
- Historical data exploration
- Flood risk assessment interface
- Model performance monitoring

[📹 Watch Demo Video](media-assets/videos/streamlit_app.mov)

---

## 3. Flask Web Service
RESTful API service offering:
- Programmatic prediction endpoints
- Batch processing capabilities
- Model metadata access
- Production-grade error handling

---

### Web Application Deployment

For **Streamlit** App, it's better to directly run it via Google Colab Notebook

For **Flask**, download the flask_app folder, create a Virtual Environmennt, install the **requirements.txt**, and run the below command:
```bash
python app.py
```
For **FastAPI**, please see the above section.

---


## -- Technical Architecture --

### Hybrid Modeling Strategy
The system implements a sophisticated dual-approach architecture:

**Temperature Prediction**: Advanced feature engineering with 67 engineered features achieving exceptional accuracy (R² = 0.992)
**Rainfall Prediction**: Optimized simple feature approach ensuring robust performance (R² = 0.716) after comprehensive experimentation

This hybrid strategy was developed through iterative experimentation, where initial complex approaches for rainfall modeling yielded suboptimal results, leading to the successful fallback strategy while maintaining advanced methodologies for temperature prediction.

### Advanced Feature Engineering Framework

The system employs a comprehensive 67-feature engineering pipeline with an 81.48% safety ratio for data leakage prevention:

**Temporal Features**: Year normalization, cyclical month encoding, seasonal decomposition
**Lag Features**: Multi-horizon historical values (1, 2, 3, 6, 12 months)
**Rolling Statistics**: Adaptive window moving averages and volatility measures
**Meteorological Features**: Rainfall-temperature interactions, anomaly detection, extreme event indicators
**Seasonal Intelligence**: Pakistan-specific monsoon patterns, winter precipitation cycles

### Conservative Feature Selection Protocol
The feature selection process implements ensemble-based validation with multiple cross-validation layers to ensure production stability and prevent overfitting in the Pakistan-specific climate context.

## -- Model Development and Training --

### Three-Phase Adaptive Training Strategy

**Phase 1**: Advanced feature engineering applied to both rainfall and temperature targets
**Phase 2**: Advanced algorithms with complex features for rainfall modeling (yielded suboptimal results)
**Phase 3**: Strategic fallback to optimized simple approach for rainfall while maintaining advanced features for temperature

This adaptive methodology demonstrates systematic model development with empirical validation at each stage.

### Algorithm Portfolio
- `Linear Regression` (baseline performance benchmarking)
- `Ridge Regression` (L2 regularization for stability)
- `Random Forest` (ensemble robustness)
- `Gradient Boosting` (sequential learning optimization)
- `XGBoost` (gradient boosting with advanced regularization)
- `Support Vector Regression` (kernel-based non-linear modeling)

### Production Model Performance
- **Temperature Model**: R² = 0.992, demonstrating exceptional predictive accuracy
- **Rainfall Model**: R² = 0.716, providing reliable forecasting for flood risk assessment
- **Cross-validation**: Robust performance across multiple validation folds
- **Hyperparameter Optimization**: GridSearchCV with systematic parameter tuning

## -- Flood Risk Analysis Framework --

### Pakistan Monsoon Pattern Analysis
The system incorporates specific analysis of Pakistan's July-August flood risk periods, utilizing:
- Coefficient of variation analysis for extreme event detection
- Monsoon intensity pattern recognition
- Historical flood correlation analysis
- Seasonal vulnerability assessment

![Flood_Risk_Analysis](media-assets/images/flood_risk_analysis.png)

### Climate Correlation Insights
**Pearson Correlation**: 0.2031 between rainfall and temperature
**Statistical Significance**: Significant but weak correlation reflecting Pakistan's complex meteorological dynamics
**Interpretation**: Independent climate processes requiring specialized modeling approaches for each target variable

## -- Data Visualization and Analysis --

### Exploratory Data Analysis
Comprehensive visualization suite covering:
- 116-year climate trend analysis
- Seasonal pattern decomposition
- Extreme weather event identification
- Regional climate variability assessment

![Data_Visualization](media-assets/images/Data_Visualization.png)

### Model Performance Visualization
- Cross-validation performance metrics
- Residual analysis and error distribution
- Feature importance ranking
- Prediction accuracy across different seasons

![Model_Evaluation](media-assets/images/model_evaluation.png)

### Interactive Analysis Dashboard
- Real-time prediction interface
- Historical pattern exploration
- Flood risk assessment tools
- Seasonal forecasting capabilities

## -- Model Evaluation and Validation --

![Model_Evaluation](media-assets/images/flask_app.png)

### Model Serialization
- **joblib format**: Optimized for sklearn-based models
- **pickle format**: Cross-platform compatibility
- **Metadata storage**: Model performance and configuration details
- **Inference functions**: Standalone prediction capabilities

### Performance Metrics
- **Root Mean Square Error (RMSE)**: Quantitative accuracy assessment
- **Mean Absolute Error (MAE)**: Absolute prediction deviation
- **R-squared (R²)**: Variance explanation capability
- **Cross-validation scores**: Generalization performance validation

### Validation Strategy
- **Time-series cross-validation**: Temporal data integrity preservation
- **Seasonal stratification**: Performance consistency across Pakistan's climate seasons
- **Holdout testing**: Final model validation on unseen data
- **Statistical significance testing**: Confidence interval analysis

---

## -- System Limitations and Considerations --

### Data Scope Limitations
- **Temporal Range**: Historical data limited to 1901-2016 period
- **Spatial Resolution**: National average data may not capture regional variations
- **Climate Change**: Recent climate shifts may not be fully represented

### Model Constraints
- **Feature Dependency**: Prediction accuracy depends on input feature availability
- **Uncertainty Quantification**: Statistical predictions with inherent confidence intervals
- **External Factors**: Limited incorporation of global climate indices

### Production Considerations
- **Model Monitoring**: Regular performance evaluation recommended
- **Data Pipeline**: Automated data quality validation protocols
- **Version Control**: Model versioning and rollback capabilities

---

## -- Future Enhancement Roadmap --

### Technical Improvements
- **Deep Learning Integration**: LSTM networks for sequential pattern recognition
- **Ensemble Methods**: Multi-model prediction averaging
- **Feature Expansion**: Integration of additional meteorological variables
- **Real-time Data**: Live weather API integration

### Operational Enhancements
- **Regional Models**: Province-level prediction capabilities
- **Alert Systems**: Automated flood risk notifications
- **Mobile Interface**: Responsive design for field applications

---

## Project Structure

```
pakistan_temp_rainfall_predictive_modelling/
├── .github/
│    └── workflows/
│        └── test.yml                            # CI/CD workflow test file
├── Flask_app/                                   # Production Flask API service
│    ├── app.py                                
│    ├── requirements.txt                     
│    ├── rainfall_model_pipeline.joblib        
│    ├── temperature_model_pipeline.joblib  
│    ├── model_metadata.joblib                   
│    ├── smart_inference_function.joblib          
│    └── templates/
│        └── index.html                          # Web interface template
├── Streamlit_app/                                
│    └── streamlit_app.py                        # Streamlit application
├── media-assets                                 # Visualization assets
│    ├── images/                                   
│    │    ├── Data_Visualization.png                   
│    │    ├── advanced_visualization.png               
│    │    ├── climate_EDA.png                          
│    │    ├── flask_app.png                            
│    │    ├── flood_risk_analysis.png                  
│    │    └── model_evaluation.png
├──  └── videos/                                      
│         └── streamlit_app.mov
├── research/
│    ├── dataset_info.md
│    ├── flood_risk_analysis_report/              # Analysis reports
│    │    └── Pakistan_Weather_Report_20250723_0606.pdf        
│    └──   pakistan_climate_data_analysis_and_predictive_modelling_ahsan_javed.ipynb  # Main analysis and training notebook
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md                                   
├── fastapi_app.py
├── model_metadata.joblib
├── rainfall_model_pipeline.joblib
├── requirements.txt
├── smart_inference_function.joblib
└── temperature_model_pipeline.joblib
```

---

**Data Source**: CHISEL @ LUMS (Center for Climate Research and Development) 
[Chisel_website](https://opendata.com.pk/organization/chisel)

**Analysis Period**: 1901-2016 Pakistan Climate Dataset
**Development Timeline**: Comprehensive iterative development with empirical validation

---

## Contact and Collaboration

For technical inquiries, model improvements, or collaboration opportunities regarding Pakistan's climate prediction capabilities, please contact through the project repository or professional networks.

**Author:**

**Ahsan Javed** _Data Scientist & ML Engineer_
- **Linkedin:** [Linkedin_link](https://www.linkedin.com/in/ahsan-javed17)
- **Email:** ahsan.javed1702@gmail.com

---

**Disclaimer**: This system is designed for research and decision support purposes. For critical flood management decisions, integrate predictions with official meteorological services and consider model limitations within operational contexts.

**Research Impact**: Contributing to Pakistan's climate resilience through advanced predictive analytics and flood risk assessment capabilities during critical monsoon periods.

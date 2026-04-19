# Traffic Congestion Prediction System Report

## 1. Introduction

Traffic congestion prediction is an important smart-city problem because it impacts commuter time, fuel consumption, logistics planning, emergency response, and pollution. This project develops a machine learning system that predicts traffic travel time and traffic congestion level using traffic, weather, time, and location signals.

## 2. Problem Definition

The objective is to build a system that can:

- Predict travel time in minutes
- Classify congestion as `Low`, `Medium`, or `High`
- Forecast future congestion trends from historical time-series data
- Expose the predictions using a backend API and a dashboard

## 3. Dataset Description

The project uses a realistic synthetic dataset to make the repository fully runnable without external dependencies. The dataset contains:

- Timestamp information
- Area name and geo-coordinates
- Temperature, humidity, rainfall, and weather condition
- Traffic volume, speed, and incident counts
- Travel time and congestion level labels

Synthetic generation follows realistic commuting behavior:

- Morning and evening rush hours increase volume
- Weekends and holidays reduce commercial traffic
- Rain and incidents lower average speed
- Different areas have different route lengths and traffic baselines

The raw dataset intentionally includes missing values and duplicate rows to demonstrate real-world cleaning.

## 4. Methodology

### 4.1 Data Preprocessing

Steps applied:

- Remove duplicate rows
- Convert numeric fields and handle nulls using area-level median
- Fill missing categorical values with mode
- Extract temporal attributes such as hour, day of week, and month
- Engineer domain features including rush-hour flag, weekend flag, holiday flag, time band, and congestion index

### 4.2 Exploratory Data Analysis

The project generates:

- Correlation heatmap
- Hour-wise average travel-time trend
- Area-vs-hour congestion heatmap
- Feature importance from the trained model

### 4.3 Model Training

Regression models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor or Gradient Boosting Regressor fallback

Classification models:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier or Gradient Boosting Classifier fallback

The train-test split is 80:20. Performance metrics include RMSE and MAE for regression, and Accuracy and weighted F1-score for classification.

### 4.4 Hyperparameter Tuning

RandomizedSearchCV is applied to the strongest baseline family using a compact search space. This improves the chosen model while keeping project runtime practical for college demonstrations.

### 4.5 Time-Series Forecasting

The mean hourly travel time is modeled using ARIMA. The system forecasts the next 24 hours and maps predicted travel time to congestion labels.

## 5. System Design

The project now follows a simpler, flatter architecture:

- `traffic_pipeline.py` contains data generation, preprocessing, modeling, and forecasting
- `main.py` exposes the FastAPI endpoints
- `dashboard/` contains the Streamlit frontend
- `spark/` contains the scalable batch-processing job

This separation improves maintainability, testing, and deployment readiness.

## 6. Big Data Considerations

To address the DSBDA angle, a PySpark job is included for large-scale processing. Spark is suitable when:

- Data size exceeds single-machine memory
- Daily batch aggregation is needed
- Data comes from multiple city zones or sensors
- Dashboard queries require pre-aggregated analytical views

The Spark pipeline reads CSV, derives temporal fields, performs area-hour aggregation, and stores output in Parquet format.

## 7. Results

The project stores model comparison metrics in `artifacts/models/training_metrics.json`. In general, ensemble models outperform linear baselines because congestion is influenced by nonlinear relationships among rainfall, incidents, rush hours, and area-specific behavior.

Feature importance generally highlights:

- Traffic volume
- Average speed
- Congestion index
- Rush hour
- Precipitation

These align with intuitive domain expectations, which supports trust in the model.

## 8. Deployment Strategy

The system can be deployed:

- Locally for demonstration
- In Docker containers for reproducibility
- On Render or AWS for online hosting

The backend and dashboard can scale independently, while heavy ETL is moved to Spark-based batch jobs.

## 9. Conclusion

This project demonstrates a practical end-to-end traffic analytics system with data engineering, machine learning, time-series forecasting, API development, dashboarding, and scalable processing. It is suitable for final-year engineering submission because it combines academic breadth with software-engineering depth.

## 10. Future Scope

- Connect to live traffic and weather APIs
- Add route optimization recommendations
- Use deep learning sequence models
- Add MLOps features such as model registry, drift tracking, and scheduled retraining

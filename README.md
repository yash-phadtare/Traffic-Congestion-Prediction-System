# Traffic Congestion Prediction System

A simplified end-to-end project for predicting traffic congestion level and travel time using time, weather, traffic volume, and area-based features.

## Simple Structure

```text
traffic-congestion-prediction/
|-- main.py                   # FastAPI app
|-- traffic_pipeline.py       # Data generation, preprocessing, training, forecasting
|-- dashboard/
|   |-- streamlit_app.py      # Streamlit dashboard
|-- scripts/
|   |-- bootstrap_project.py  # Generates data and model artifacts
|-- spark/
|   |-- traffic_spark_job.py  # Optional PySpark batch aggregation
|-- data/
|   |-- raw/
|   |-- processed/
|-- artifacts/
|   |-- models/
|   |-- plots/
|   |-- forecasts/
|-- docs/
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
```

## What It Does

- Generates a realistic traffic dataset
- Cleans and preprocesses the data
- Trains regression and classification models
- Forecasts future congestion using ARIMA
- Serves predictions through FastAPI
- Shows trends and predictions in Streamlit
- Includes an optional PySpark job for the big-data angle

## Main Files

- [main.py](</d:/DSBDA MP/main.py>)
- [traffic_pipeline.py](</d:/DSBDA MP/traffic_pipeline.py>)
- [dashboard/streamlit_app.py](</d:/DSBDA MP/dashboard/streamlit_app.py>)
- [scripts/bootstrap_project.py](</d:/DSBDA MP/scripts/bootstrap_project.py>)
- [spark/traffic_spark_job.py](</d:/DSBDA MP/spark/traffic_spark_job.py>)

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate data, train models, and create plots:

```bash
python -m scripts.bootstrap_project
```

Run the API:

```bash
python -m uvicorn main:app --reload
```

Run the dashboard in another terminal:

```bash
python -m streamlit run dashboard/streamlit_app.py
```

## API Endpoints

- `GET /health`
- `POST /predict`
- `POST /train`
- `GET /data`
- `GET /forecast`
- `GET /simulate`

## Docker

```bash
docker compose up --build
```

## Notes

- The dataset is synthetic but realistic enough for demos and academic submission.
- If `pyspark` is not installed, the Spark step is skipped during bootstrap.
- Generated files are saved inside `data/` and `artifacts/`.

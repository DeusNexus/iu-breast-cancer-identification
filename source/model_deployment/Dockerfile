# Building docker image for prediction API

# Base Image
FROM python:3.10-slim-buster

# Install necessary system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libblas3 \
        liblapack3 \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py

# Make dir for html that will be the front-end for FastAPI
RUN mkdir /app/html
COPY html/* /app/html/

# Make a dir and copy the best model (balanced dataset 2) which will be used for predictions
COPY train_logistic_reg_best_model_2024-02-29.joblib app/train_logistic_reg_best_model_2024-02-29.joblib
COPY dataset.csv app/dataset.csv

# Set the workdir to app
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Make sure we can access the Docker Container from the local host machine using 127.0.0.1:8000
EXPOSE 8000

# Exposing to 0.0.0.0 is important to make it available
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
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
RUN mkdir app/models
RUN mkdir app/models/best_model
COPY best_model_architecture.json app/models/best_model/best_model_architecture.json
COPY best_weights_balanced-2_dataset.h5 app/models/best_model/best_weights_balanced-2_dataset.h5

# Set the workdir to app
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Make sure we can access the Docker Container from the local host machine using 127.0.0.1:8000
EXPOSE 8000

# Exposing to 0.0.0.0 is important to make it available
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
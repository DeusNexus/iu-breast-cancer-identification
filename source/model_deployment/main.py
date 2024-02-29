# Importing necessary libraries and modules
import os
import sklearn
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from pydantic import BaseModel
from joblib import load
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_TO_SERVE = 'train_logistic_reg_best_model_2024-02-29.joblib'

# Load model
model = load(MODEL_TO_SERVE)

print(model)

# Define file paths for HTML, CSS, and favicon files
html_file_path = Path(__file__).parent / "html" / "index.html"
css_file_path = Path(__file__).parent / "html" / "index.css"
favicon_file_path = Path(__file__).parent / "html" / "favicon.ico"

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for your data
class Features(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

##############################
# Load current dataset from file, but in future this should be from database
# Load the dataset
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Removing Empty Column
df.drop(columns=['id','Unnamed: 32'],inplace=True)

# Move diagnosis to be last column
df['diagnosis'] = df.pop('diagnosis')

# Prepare X_train, X_test, y_train, y_test
X = df.drop(columns='diagnosis')
y = df['diagnosis']
################################

##########################################################
##### PCA FIGURE
# Assuming 'B' and 'M' are represented as 0 and 1 in the target variable 'y'
# Step 1: Apply PCA to reduce the features to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 2: Make predictions using the trained model
predictions = model.predict(X)

# Adjust the conditions to match string representations
benign_indices = (y == 'B')
malignant_indices = (y == 'M')

# Ensure that correct and incorrect predictions are computed based on the actual values
correct_predictions = (predictions == y.values)
incorrect_predictions = ~correct_predictions
###########################################################

# Enable CORS for all origins (useful for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to serve the index.html file
@app.get("/")
def read_root():
    return FileResponse(html_file_path, media_type="text/html")

# Endpoint to serve the index.css file
@app.get("/index.css")
def get_css():
    return FileResponse(css_file_path, media_type="stylesheet")

# Endpoint to serve the favicon.ico file
@app.get("/favicon.ico")
def get_favicon():
    return FileResponse(favicon_file_path, media_type="image/x-icon")

# Endpoint to predict emotion based on the uploaded image
@app.post("/api/predict")
async def predict(features: Features):
    try:
        # Convert the Pydantic model to a numpy array
        feature_values = [features.model_dump()[name] for name in features.model_dump()]
        features_array = np.array(feature_values).reshape(1, -1)

        # Transform the new data point with PCA
        new_point_pca = pca.transform(features_array)

        # Ensure the input is in the same format as during training, you might need to convert it to a DataFrame
        # features_df = pd.DataFrame([feature_values], columns=features.dict().keys())
        
        # Make predictions with your model
        prediction = model.predict(features_array)
        predicted_label = prediction[0]

        # If your model provides probabilities
        probabilities = model.predict_proba(features_array)[0]

        # Now, let's adjust the plotting code accordingly
        plt.figure(figsize=(15, 15))

        # Benign correct (black circle)
        plt.scatter(X_pca[benign_indices & correct_predictions, 0], X_pca[benign_indices & correct_predictions, 1], c='orange', marker='o', label='Benign Correct')

        # Malignant correct (black triangle)
        plt.scatter(X_pca[malignant_indices & correct_predictions, 0], X_pca[malignant_indices & correct_predictions, 1], c='lightblue', marker='o', label='Malignant Correct')

        # Benign incorrect (red circle)
        plt.scatter(X_pca[benign_indices & incorrect_predictions, 0], X_pca[benign_indices & incorrect_predictions, 1], c='black', marker='^', label='Benign Incorrect')

        # Malignant incorrect (red triangle)
        plt.scatter(X_pca[malignant_indices & incorrect_predictions, 0], X_pca[malignant_indices & incorrect_predictions, 1], c='black', marker='x', label='Malignant Incorrect')

        # Plot the new data point on the PCA plot
        plt.scatter(new_point_pca[:, 0], new_point_pca[:, 1], c='blue', label='New Data Point', marker='o')

        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('PCA of Predictions with Class Distinctions')
        plt.tight_layout()
        plt.legend()

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Encode the image in memory as base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        ### END PCA FIGURE


        b_prob, m_prob = probabilities

        # Create a response dictionary
        response_dict = {
            "predicted_class": predicted_label,
            "probabilities": f'B: {round(b_prob * 100,4)}% \nM: {round(m_prob * 100,4)}%',
            "image": image_base64
        }

        # Return predictions in JSON format
        return JSONResponse(content=response_dict, status_code=200)
    
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

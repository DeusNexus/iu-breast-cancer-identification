# Importing necessary libraries and modules
import os
from io import BytesIO
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json

# Set TensorFlow logging level to suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or '3' to additionally suppress all warnings

# Load pre-trained Keras model architecture and weights from JSON file and H5 file, respectively
with open("./models/best_model/best_model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()

keras_best_model = model_from_json(loaded_model_json)

# keras_best_model.load_weights("./models/best_model/best_weights.h5")
# keras_best_model.load_weights("./models/best_model/best_weights_balanced_dataset.h5")
keras_best_model.load_weights("./models/best_model/best_weights_balanced-2_dataset.h5")

# Define file paths for HTML, CSS, and favicon files
html_file_path = Path(__file__).parent / "html" / "index.html"
css_file_path = Path(__file__).parent / "html" / "index.css"
favicon_file_path = Path(__file__).parent / "html" / "favicon.ico"

# Initialize FastAPI app
app = FastAPI()

print('API Created')

# Enable CORS for all origins (useful for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print('Middleware Created')

# Define emotion labels
labels = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

# Utility function to preprocess the image
def preprocess_image(file):
    # Read the content of the file into a BytesIO object
    file_content = BytesIO(file.read())

    # Load and preprocess the image using the same approach as in your testing code
    img = image.load_img(file_content, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1] range

    return img_array

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
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Preprocess the image
        img_array = preprocess_image(file.file)
        
        # Make predictions
        predictions = keras_best_model.predict(img_array)

        # Get the predicted emotion (assuming your model has a softmax output layer)
        predicted_emotion = int(np.argmax(predictions))

        # Create a dictionary of class probabilities
        class_probabilities_dict = []
        for idx, prob in enumerate(predictions[0]):
            class_probabilities_dict.append({ labels[idx]: prob })
        class_probabilities = str({ key: value for item in class_probabilities_dict for key, value in item.items() })

        # Return predictions in JSON format
        return JSONResponse(content={
            "predicted_emotion": predicted_emotion,
            "predicted_emotion_label": labels[int(predicted_emotion)],
            "class_probabilities": class_probabilities,
            "label_encodings": labels
        }, status_code=200)
    
    except Exception as e:
        # Handle exceptions and return an Internal Server Error if necessary
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
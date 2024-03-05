## Front-end of Docker API
The trained model uses the Logistic Classifier that performed the best according to all the collected metrics and results.
- The user can select one of the example data to be predicted by the model or fill out their own numeric data.
- The user clicks predict.
- The predicted class (B)enign or (M)alignant, class probabilty and how it relates to the trained model predictions on the PCA1 and PCA2 is displayed.
- The PCA plot gives great insight how the newly classified datapoint relates and if it is close to the incorrectly predicted ones or where the two classes are difficult to distinguish extra care can be taken.

**Deploy Docker-api Front-end**
<img src="/docs/deploy_model_docker_api.png" alt="Deploy Docker-api front-end" width="750">

# Pull the pre-build image for [Docker Hub](https://hub.docker.com/repository/docker/deusnexus/breast_cancer_classification/general)
### Pull the latest build image
`docker pull deusnexus/breast_cancer_classification:latest`
### Run the container
`docker run --name breast_cancer_classification -p 8000:8000 deusnexus/breast_cancer_classification:latest`
### Open the API on localhost
`http://127.0.0.1:8000`

# Building Image
### Enter docker folder
`cd docker-api`
### Build the image
`docker build -t breast_cancer_classification:latest .`
### Run the container
`docker run --name breast_cancer_classification -p 8000:8000 breast_cancer_classification:latest`
### Open the API Front-end
`http://127.0.0.1:8000`

# Running API directly
### Enter source/model_deployment directory
`cd source/model_deployment`
### Create a local pythong venv
`python3 -m venv venv`
### Activate the virtual environment
`source venv/bin/activate`
### Install the required modules
`pip3 install -r requirements.txt`
### Run the API 
`uvicorn main:app --reload`
### Open the API Front-end
`http://127.0.0.1:8000`
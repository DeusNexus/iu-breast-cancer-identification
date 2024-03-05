# IU INTERNATIONAL UNIVERSITY OF APPLIED SCIENCES
### Model Engineering (DLBDSME01) - **Task 1: Development of an Interpretable Model for Breast Cancer Prediction**

Abstract: 
....

### Purpose
According to the breast cancer is one of the most common cancer types overall. Diagnosing a breast pathology in time is very important since it significantly increases the chances of successful treatment and survival. Thus, a fast and accurate classification of a tumor as benign or malignant is important. Moreover, to increase technology acceptance, the trustworthiness of the approach is critical.

A group of oncologists have seen impressive breast cancer identification results from an AI algorithm on a fair. However, the group did not understand why the model predicted certain tumors as malign and others as benign - the validity of the model was questioned in some cases. The group also discussed the most important features for the predictions. Finally, they decided to ask a group of data scientists, you are one of them, for possibilities to understand the prediction of a possible machine learning model. Your task is to develop a classification model that predicts a tumor to be malignant or benign with high accuracy (F1 score > 0.95). Moreover, the model should be interpretable. The outcomes should be presented/communicated to non-experts to convince them about the trustworthiness of the chosen approach.

### Task
`Support a group of oncologists with the interpretable prediction model to allow for additional indications that can be produced automatically as well as support understanding to ease technology acceptance.`

# Project Organization
`Organize the project using the CRISP-DM. Make a proposal on how to build the folder structure of a Git repository for the project.`

**Git Project Structure**
<img src="/docs/model_engineer_folder_structure.png" alt="Git Project Structure" width="750">

The project will be organized using the CRISP-DM (Cross-Industry Standard Process for Data Mining) method. This method provides a structured approach to organizing and executing data science projects. The proposed Git repository structure is as follows:

- **data/original_data**: This folder will contain the dataset and any additional data files.
  - `dataset.csv`: File with a list of IDs, labels, and features for breast cancer classification.
  - `Addition_Information_Case_Study_Task_1.pdf`: File will information about the `dataset.csv` (columns, attributes, further info)
- **data/processed_data**:
  - Data that is processed is stored in central database MongoDB under the `processed_data` collection along with metadata for versioning and how the data has been processed.
  - This folder is purely for staging and should not really be used to store data. For sake of simplicity data in the notebooks is not loaded from the database but the general concept and metadata is still explained and given.

- **docs**: Documentation related to the projec including the necessary created documents and media created during the training, evaluation, deployment and additional for clarifying the process.
  - `README.md`: Main project documentation.
  - `project_report.pdf`: Final project report summarizing findings and decisions.
  - `media`: Images

- **notebooks**: This folder will include Jupyter notebooks used for data exploration.
  - `exploratory_data_analysis.ipynb`: Notebook for exploring the dataset.

- **scripts**: Any supporting scripts used in the project to processed data or interact with database.
  - `database.py`: Contains all methods to connect and insert original or processed data to the MongoDB database collections `original_data` and `processed_data`.
  - `write_original_data.py`: Python script commiting the original data to the database with metadata enrichment for data versioning.
  - `write_processed_data.py`: Python script for commiting processed data to the database with metadata and include the data processing steps.
  - `README.md`: For for clarification.

- **source**: Contains the model_training, model_evaluation and model_deployment steps of CRISP-DM.
  - **source/model_training**:
    - `model_training.ipynb`: Notebook for training different candidate models.
    - `train_logistic_reg_best_model_2024-02-29.joblib`: Trained model using LogisticRegression model from scikit-learn.
    - `train_model_best_cv_nsplit_accuracy.csv`: Training artifact to see best cross-validation splits using accuracy metric.
    - `train_model_result_metrics.csv`: All trained models with their individual performances and their interpretability:
      - Model, Interpretability, F1 Score, ROC AUC, Recall, Precision, Accuracy (Testing), Accuracy (Training)
  - **source/model_evaluation**:
    - `model_evaluation.ipynb`: Notebook for evaluating model performance.
    - `evaluate_best_model_feature_importance_standardized_vs_regular_ci.csv`: Evaluation artifact containing the most important features, their ci error interval, standardized and non-standardized values.
  - **source/model_deployment**: Folder for model deployment, including `Dockerfile` for building an image.
    - `/html`: Folder with front-end index.html file and css.
    - `dataset.csv`: Used by the Dockerfile make the current dataset available (used for predictions, PCA1 & PCA2 mappings)
    - `main.py`: File used by FastAPI
    - `requirements.txt`: Python module requirements to be installed in the Docker image.
    - `train_logistic_reg_best_model_2024-02-29.joblib`: Saved LogisticRegression model that is loaded in the Docker image for predictions.
    - `README.md`: Additional information on Docker Image - Use, Building, etc.
- `.gitignore`: File to ignore github uploads of venv, venv-docker, __pycache__
- `LICENSE`: License file
- `requirements.txt`: Requirements file to run the notebooks.


##################
##################
*** At the# end of each step of this use case, critically assess whether all necessary operations have been conducted and provide justifications for the decisions made during the process. ***
##################
##################

# Dataset Quality Evaluation
`Assess the quality of the provided dataset by performing the following tasks:`

- **Exploratory Data Analysis (EDA)**: Conduct a thorough exploration of the dataset, examining statistical summaries, distributions, and relationships between variables.

- **Data Cleaning**: Check for missing values, outliers, and inconsistencies. Clearly document any data cleaning steps taken.

- **Visualizations**: Create visualizations to effectively communicate key relationships and patterns within the data, ensuring business partners can understand important aspects easily.

# Candidate Models
`Establish a collection of candidate models considering the task requirements. Focus on increasing complexity gradually while prioritizing interpretability. Key steps include:`

- **Model Training**: Train multiple models with varying complexities (e.g., logistic regression, decision trees, ensemble methods). Document the training process and hyperparameter choices.

- **Model Evaluations**: Evaluate the performance of each model using appropriate metrics, ensuring high accuracy (F1 score > 0.95). Provide detailed explanations of the evaluation results.

- **Best Model**: Select the best-performing model based on the evaluation metrics and feature importance analysis.

## Interpretability
`Ensure the developed model is interpretable:`

- Utilize models with inherent interpretability, such as decision trees or linear models.
- Provide clear explanations of the model's decision-making process.
- Use visualizations to illustrate important aspects of the model's behavior.

## Importance of Explanatory Variables
`Discuss the importance of each explanatory variable:`

- Conduct feature importance analysis to identify variables contributing significantly to model predictions.
- Clearly communicate the relevance of each variable in the context of breast cancer prediction.

## Detailed Error Analysis
`Perform a detailed error analysis to understand the weaknesses of the approach:`

**Confusion Matrix**
<img src="/docs/evaluate_best_model_logistic_confusion_matrix.jpg" alt="Evaluate Confusion matrix" width="500">

**Feature Importance Rank**
<img src="/docs/evaluate_best_model_logistic_feature_importance_rank.jpg" alt="Evaluate Feature importance rank" width="500">

**PCA Predict Correct-Incorrect**
<img src="/docs/evaluate_best_model_logistic_PCA_predictions_correct_incorrect.jpg" alt="Evaluate PCA predict correct incorrect" width="500">

**ROC Curve**
<img src="/docs/evaluate_best_model_logistic_roc_curve.jpg" alt="Evaluate ROC Curve" width="500">


- Identify common types of errors made by the model (e.g., false positives, false negatives).
- Analyze specific instances where the model failed to make accurate predictions.
- Provide insights into potential improvements or areas for further investigation.

# Making the Model Accessible for Operations
`Propose how the developed model can be integrated into daily work:`

- Design a graphical user interface (GUI) for easy interaction with the model.
- Outline the steps for incorporating the model into the workflow of doctors.
- Consider user-friendly features to enhance the acceptance and usability of the model.

## Front-end of Docker API
### See more information in [source/model_deployment/README.md](./source/model_deployment/README.md)
The trained model uses the Logistic Classifier that performed the best according to all the collected metrics and results.
- The user can select one of the example data to be predicted by the model or fill out their own numeric data.
- The user clicks predict.
- The predicted class (B)enign or (M)alignant, class probabilty and how it relates to the trained model predictions on the PCA1 and PCA2 is displayed.
- The PCA plot gives great insight how the newly classified datapoint relates and if it is close to the incorrectly predicted ones or where the two classes are difficult to distinguish extra care can be taken.

**Docker-api Front-end**
<img src="/docs/deploy_model_docker_api.png" alt="Deploy Docker-api front-end" width="750">

## How to get started
### Dependencies for using the Jupyter Notebooks
Create a new virtual python environment for the notebooks.

`python3 -m venv venv`

Activate the environment (Linux)

`source venv/bin/activate`

Install the dependencies

`pip3 install -r requirements.txt`

## API usage using pre-build docker image
### Pull the latest build image
`docker pull deusnexus/breast_cancer_classification:latest` 
### Run the container
`docker run --name breast_cancer_classification -p 8000:8000 deusnexus/breast_cancer_classification:latest`
### Open the API on localhost
`http://127.0.0.1:8000`

# Reflection
...

# Conclusion
...

# Disclaimer
The developed application is licensed under the GNU General Public License.

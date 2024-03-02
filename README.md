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
`Organize the project using the CRISP-DM or the MS Team Data Science method. Make a proposal on how to build the folder structure of a Git repository for the project.`

The project will be organized using the CRISP-DM (Cross-Industry Standard Process for Data Mining) method. This method provides a structured approach to organizing and executing data science projects. The proposed Git repository structure is as follows:

- **data**: This folder will contain the dataset and any additional data files.
  - `data.csv`: File with a list of IDs, labels, and features for breast cancer classification.

- **notebooks**: This folder will include Jupyter notebooks used for data exploration, preprocessing, model training, and evaluation.
  - `1_data_exploration.ipynb`: Notebook for exploring the dataset.
  - `2_data_preprocessing.ipynb`: Notebook for data cleaning and feature engineering.
  - `3_model_training.ipynb`: Notebook for training different candidate models.
  - `4_model_evaluation.ipynb`: Notebook for evaluating model performance.
  - `5_interpretability_analysis.ipynb`: Notebook focusing on interpretability aspects.

- **models**: This folder will store the trained models.
  - `model1.pkl`: Trained model using [algorithm 1].
  - `model2.pkl`: Trained model using [algorithm 2].
  - ...

- **visualizations**: This folder will contain visualizations generated during data exploration and model evaluation.
  - `feature_importance.png`: Visualization of feature importance.
  - `error_analysis.png`: Visualization of detailed error analysis.

- **docs**: Documentation related to the project.
  - `README.md`: Main project documentation.
  - `project_report.pdf`: Final project report summarizing findings and decisions.

- **scripts**: Any supporting scripts used in the project.
  - `preprocessing.py`: Python script for data preprocessing.
  - `model_evaluation.py`: Script for evaluating model performance.

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

- Identify common types of errors made by the model (e.g., false positives, false negatives).
- Analyze specific instances where the model failed to make accurate predictions.
- Provide insights into potential improvements or areas for further investigation.

# Making the Model Accessible for Operations
`Propose how the developed model can be integrated into daily work:`

- Design a graphical user interface (GUI) for easy interaction with the model.
- Outline the steps for incorporating the model into the workflow of doctors.
- Consider user-friendly features to enhance the acceptance and usability of the model.

*** At the end of each step of this use case, critically assess whether all necessary operations have been conducted and provide justifications for the decisions made during the process. ***

# Front-end of Docker API
### See more information in source/model_deployment/README.md
The trained model uses the Logistic Classifier that performed the best according to all the collected metrics and results.
- The user can select one of the example data to be predicted by the model or fill out their own numeric data.
- The user clicks predict.
- The predicted class (B)enign or (M)alignant, class probabilty and how it relates to the trained model predictions on the PCA1 and PCA2 is displayed.
- The PCA plot gives great insight how the newly classified datapoint relates and if it is close to the incorrectly predicted ones or where the two classes are difficult to distinguish extra care can be taken.
![docker-api front-end](/docs/docker_api.png)

# How to get started
## Dependencies for using the Jupyter Notebooks
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

## Reflection
...

## Conclusion
...

# Disclaimer
The developed application is licensed under the GNU General Public License.

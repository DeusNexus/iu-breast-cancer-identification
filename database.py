from pymongo import MongoClient
from pymongo.errors import ValidationError
from datetime import datetime
from typing import Dict, List, Union

def connect_db(
        url='mongodb://localhost:27017/',
        db_name='breast_cancer_mlops'
    ):
    """
    Establishes a connection to the MongoDB database.
    
    Parameters:
    - url (str): MongoDB connection URL.
    - db_name (str): Name of the database to connect to.
    
    Returns:
    - db: A reference to the specified MongoDB database.
    """
    # Establish a connection to the MongoDB server
    client = MongoClient(url)

    # Select your database
    db = client[db_name]

    return db

def close_db(db):
    """
    Closes the connection to the MongoDB database.
    
    Parameters:
    - db: The database connection to close.
    
    Returns:
    - None
    """
    # Close client connection to database
    return db.close()

def insert_raw_document(
        db: MongoClient, 
        data_id: Union[int, str], 
        features: Dict[str, Union[int, float, str]], 
        collection_date: datetime, 
        source: str, 
        dataset_label: str
    ) -> bool:
    
    """
    Inserts a new document into the raw data collection.
    
    Parameters:
    - db: The database connection.
    - data_id (str): Unique identifier for the data.
    - features (dict): Feature values of the data.
    - collection_date (str): Date when the data was collected.
    - source (str): Source of the data.
    - dataset_label (str): Label identifying the dataset.
    
    Returns:
    - bool: True if insertion was successful, False otherwise.
    """
    # Example:
    # insert_raw_document(
    #     db, 
    #     data_id='12345', 
    #     features={}, 
    #     collection_date='2024-02-25', 
    #     source='Hospital XYZ', 
    #     dataset_label='Breast_Cancer_Study_2024'
    # )

    collection = db['raw_data']

    # Example document following the raw data schema
    document = {
        "data_id": f"{data_id}", #"12345",
        "features": {
            f"{features}" # Feature values here
        },
        "metadata": {
            "collection_date": f"{collection_date}", #"2024-02-25"
            "source": f"{source}", #"Hospital XYZ"
            "dataset_label": f"{dataset_label}", #"Breast_Cancer_Study_2024"
        }
    }

    try:
        # Insert the document into the specified collection
        result = collection.insert_one(document)
        print(f"Document inserted with _id: {result.inserted_id}")
        return True
    except ValidationError as e:
        print(f"Document failed validation: {e}")
        return False

def insert_processed_document(
        db: MongoClient, 
        processed_data_id: Union[int, str], 
        original_data_id: Union[int, str], 
        features: Dict[str, Union[int, float, str]], 
        steps: List[str], 
        processed_on: datetime, 
        processed_by: str, 
        dataset_label: str, 
        comments: str
    ) -> bool:
      
    """
    Inserts a new document into the processed data collection.
    
    Parameters:
    - db: The database connection.
    - processed_data_id (str): Unique identifier for the processed data.
    - original_data_id (str): Identifier linking back to the original raw data.
    - features (dict): Processed feature values.
    - steps (list): List of processing steps applied to the data.
    - processed_on (str): Date when the data was processed.
    - processed_by (str): Identifier of the person or system that processed the data.
    - dataset_label (str): Label identifying the processed dataset.
    - comments (str): Additional comments regarding the processing.
    
    Returns:
    - bool: True if insertion was successful, False otherwise.
    """

    # Example:
    # insert_processed_document(
    #     db, 
    #     processed_data_id='67890', 
    #     original_data_id='12345', 
    #     features={}, 
    #     steps=["scaled", "encoded"], 
    #     processed_on="2024-03-01", 
    #     processed_by="DataEngineerA", 
    #     collection_date='2024-02-25', 
    #     dataset_label='Breast_Cancer_Study_2024_Processed', 
    #     comments='Data scaled and encoded for model input.'
    # )

    collection = db['processed_data']
    
    # Example document following the processed data schema
    document = {
        "processed_data_id": f"{processed_data_id}", #"67890"
        "original_data_id": f"{original_data_id}", #"12345"
        "features": f"{features}", # Processed feature values here
        "processing_details": {
            "steps": f"{steps}", #["scaled", "encoded"],
            "processed_on": f"{processed_on}", #"2024-03-01",
            "processed_by": f"{processed_by}", #"DataEngineerA"
        },
        "metadata": {
            "dataset_label": f"{dataset_label}", #"Breast_Cancer_Study_2024_Processed",
            "comments": f"{comments}", #"Data scaled and encoded for model input."
        }
    }

    try:
        # Insert the document into the specified collection
        result = collection.insert_one(document)
        print(f"Document inserted with _id: {result.inserted_id}")
        return True
    except ValidationError as e:
        print(f"Document failed validation: {e}")
        return False
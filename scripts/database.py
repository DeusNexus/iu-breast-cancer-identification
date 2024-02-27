from pymongo import MongoClient
from pymongo.errors import WriteError
from datetime import datetime
from typing import Dict, List, Union
import pandas as pd

def connect_db(
        url:str='mongodb://localhost:27017/',
        db_name:str='breast_cancer_mlops'
    ) -> MongoClient:
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

def close_db(db: MongoClient):
    """
    Closes the connection to the MongoDB database.
    
    Parameters:
    - db: The database connection to close.
    
    Returns:
    - None
    """
    # Close client connection to database
    return db.close()

def get_documents(
        db: MongoClient,
        collection: str,
        query: object
    ) -> object:

    # Retrieve all documents matching the filter from the collection
    matching_documents = db[collection].find(query)

    return matching_documents

def insert_original_document(
        db: MongoClient, 
        data_id: Union[str], 
        features: Dict[str, Union[int, float, str, None]], 
        collection_date: datetime, 
        source: str, 
        dataset_label: str
    ) -> bool:
    
    """
    Inserts a new document into the original data collection.
    
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
    # insert_original_document(
    #     db, 
    #     data_id='12345', 
    #     features={}, 
    #     collection_date='2024-02-25', 
    #     source='Hospital XYZ', 
    #     dataset_label='Breast_Cancer_Study_2024'
    # )

    collection = db['original_data']

    # Example document following the original data schema
    document = {
        "data_id": data_id, #"12345",
        "features": features, # Feature values here
        "metadata": {
            "collection_date": collection_date, #"2024-02-25"
            "source": source, #"Hospital XYZ"
            "dataset_label": dataset_label, #"Breast_Cancer_Study_2024"
        }
    }

    # Handle 'nan' and convert 'datetime' to 'string'
    for key, value in document['features'].items():
        if pd.isna(value):
            document['features'][key] = ''  # Replace 'nan' with 0 or other appropriate value

    document['metadata']['collection_date'] = document['metadata']['collection_date'].strftime('%Y-%m-%d')

    # Now you can attempt to insert the document again

    print(document)

    try:
        # Insert the document into the specified collection
        return collection.insert_one(document)
    except WriteError as e:
        print(f"Document failed validation: {e}")
        return False

def insert_processed_document(
        db: MongoClient, 
        processed_data_id: Union[str], 
        original_data_id: Union[str], 
        features: Dict[str, Union[int, float, str, None]],
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
    #     dataset_label='Breast_Cancer_Study_2024_Processed', 
    #     comments='Data scaled and encoded for model input.'
    # )

    collection = db['processed_data']
    
    # Example document following the processed data schema
    document = {
        "processed_data_id": processed_data_id, #"67890"
        "original_data_id": original_data_id, #"12345"
        "features": features, # Processed feature values here
        "processing_details": {
            "steps": steps, #["scaled", "encoded"],
            "processed_on": processed_on, #"2024-03-01",
            "processed_by": processed_by, #"DataEngineerA"
        },
        "metadata": {
            "dataset_label": dataset_label, #"Breast_Cancer_Study_2024_Processed",
            "comments": comments, #"Data scaled and encoded for model input."
        }
    }

    # Handle 'nan' and convert 'datetime' to 'string'
    for key, value in document['features'].items():
        if pd.isna(value):
            document['features'][key] = ''  # Replace 'nan' with 0 or other appropriate value

    document['processing_details']['processed_on'] = document['processing_details']['processed_on'].strftime('%Y-%m-%d')

    # Now you can attempt to insert the document again

    print(document)

    try:
        # Insert the document into the specified collection
        return collection.insert_one(document)
    except WriteError as e:
        print(f"Document failed validation: {e}")
        return False
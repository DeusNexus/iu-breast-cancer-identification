from database import connect_db, close_db, insert_original_document
from datetime import datetime
from functools import wraps
import pandas as pd
import uuid

def mongo_connection(func):
    @wraps(func)
    def with_connection(*args, **kwargs):
        db = connect_db()
        try:
            result = func(db, *args, **kwargs)
        finally:
            close_db(db.client)  # Assuming db.close() is meant to close the MongoClient
        return result
    return with_connection

@mongo_connection
def db_operations(db, *args, **kwargs):

    # Load the CSV file into a DataFrame
    df = pd.read_csv('../data/original_data/dataset.csv')
    now = datetime.now() # Current datetime
    data_uuid = uuid.uuid4() # One data_id for entire dataset

    for index, row in df.iterrows():
        data_id = str(data_uuid)
        features = row.to_dict()
        collection_date = now  # Adjust if you have a specific column for the date
        source = "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29"  # Define as needed
        dataset_label = "Breast Cancer Wisconsin Diagnostic"  # Define as needed

        # Example:
        # insert_original_document(
        #     db, 
        #     data_id='12345', 
        #     features={}, 
        #     collection_date='2024-02-25', 
        #     source='Hospital XYZ', 
        #     dataset_label='Breast_Cancer_Study_2024'
        # )

        insert_original_document(
            db=db, 
            data_id=data_id, 
            features=features, 
            collection_date=collection_date, 
            source=source, 
            dataset_label=dataset_label
        )

    pass

db_operations()
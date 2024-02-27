from database import connect_db, close_db, insert_processed_document, get_documents
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

    # Load the original data from MongoDB into a DataFrame
    # Specifiy query
    original_data_id = 'a51f37a5-ae67-4875-af9e-0d291bef262b'

    # Define the filter for the query
    query_filter = {'data_id': original_data_id}

    # Retrieve all documents matching the filter from the collection
    docs = get_documents(db,'original_data',query_filter)

    # Initialize an empty list to store the rows
    rows = []
    metadata = {}

    metadata['data_id'] = docs[0]['data_id'] # One 'raw' datasets shares one unique data_id
    metadata['collection_date'] = docs[0]['metadata']['collection_date']
    metadata['source'] = docs[0]['metadata']['source']
    metadata['dataset_label'] = docs[0]['metadata']['dataset_label']

    # Iterate over the first 10 documents and extract the features
    for doc in docs:
        # Extract the 'features' dictionary
        cols = doc['features']
        # Append the features dictionary to the rows list
        rows.append(cols)
    
    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)

    ########################################
    ############### CLEAN DATA #############
    ########################################

    # Drop empty column
    df.drop(columns='Unnamed: 32', inplace=True)

    # Drop obsolete id column
    df.drop(columns='id', inplace=True)

    # Move to last column (features first)
    df['diagnosis'] = df.pop('diagnosis')

    # Encode diagnosis to B=0 and M=1
    # Define the mapping from 'B' and 'M' to 0 and 1, respectively
    diagnosis_mapping = {'B': 0, 'M': 1}

    # Apply the mapping to the 'diagnosis' column
    df['diagnosis'] = df['diagnosis'].map(diagnosis_mapping)
    
    #########################################
    ########### END OF CLEANING #############
    #########################################
    
    # Optionally, print the DataFrame to check the output
    print(df)
    print(metadata)

    # The DataFrame 'df' is now ready for further processing or analysis
    now = datetime.now() # Current datetime
    data_uuid = uuid.uuid4() # One data_id for entire dataset

    for index, row in df.iterrows():
        data_id_processed = str(data_uuid)
        features = row.to_dict()

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

        insert_processed_document(
            db=db,
            processed_data_id=data_id_processed, 
            original_data_id=metadata['data_id'],
            features=features, 
            steps=["encoded"], 
            processed_on=now, 
            processed_by='DeusNexus',  
            dataset_label='Breast Cancer Cleaned', 
            comments='Column 32 dropped, id dropped, target variable moved to last column and encoded (B=0, M=1)'
        )

db_operations()
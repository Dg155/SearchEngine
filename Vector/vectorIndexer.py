import math
from pymilvus import DataType, MilvusClient
from towhee import AutoPipes, pipe, ops, DataCollection
import sys
import os
import pandas as pd
import numpy as np

collection_name = "DEV"

client = MilvusClient(
uri="https://in03-2893550dac2ce6c.api.gcp-us-west1.zillizcloud.com",
token="69982138f98274926e691ee7f5c44a775f816ce0065d51fbb3fa95fbdb6be466c9f2d3c603354518c8cafcce30af0b3948049dcc"
)

def create_milvus_collection(dim):
    
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    #field = ["id", "title", "title_vector", "link", "content", "content_vector"]
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
    #schema.add_field(field_name="title_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="link", datatype=DataType.VARCHAR, max_length=500)
    #schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="content_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)


    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="id"
    )

    #index_params.add_index(
    #    field_name="title_vector", 
    #    index_type="AUTOINDEX",
    #    metric_type="IP"
    #)

    index_params.add_index(
        field_name="content_vector", 
        index_type="AUTOINDEX",
        metric_type="IP"
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )




if __name__ == "__main__":


    client.drop_collection(collection_name=collection_name)

    create_milvus_collection(384)

    # Define the column names and data types according to your schema
    column_names = ['id', 'title', 'link', 'content_vector']
    column_types = {'id': 'int64', 'title': 'str', 'link': 'str', 'content_vector': 'object'}

    # Read the CSV file, skipping the first row, and apply conversion function to relevant columns
    df = pd.read_csv(r"C:\Users\kidro\OneDrive\Desktop\School\SearchEngine\Vector\fileInfoDEV.csv", skiprows=[0], names=column_names, dtype=column_types)

    def strip_vector(stringVector):
        return np.array([float(x.rstrip('\n')) for x in stringVector.strip('[]').split(' ') if x != ''], dtype= 'float32')
    
    def convert_string(string):
        string = str(string)
        return string[:400] + '...' if len(string) >= 500 else string
    
    df['content_vector'] = df['content_vector'].apply(strip_vector)
    df['link'] = df['link'].apply(convert_string)
    df['title'] = df['title'].apply(convert_string)
   
    dict_version = df.to_dict(orient='records')

    chunk_size = 1000  

    # Calculate the total number of chunks
    total_chunks = math.ceil(len(dict_version) / chunk_size)

    for chunk_index in range(total_chunks):
        start_index = chunk_index * chunk_size
        end_index = min((chunk_index + 1) * chunk_size, len(dict_version))
        chunk_data = dict_version[start_index:end_index]

        
        # insert into the vector database
        res = client.insert(
        collection_name=collection_name,
        data= chunk_data
        )

    
    print("hosting")
    exit()
    sentence_embedding = AutoPipes.pipeline('sentence_embedding')
    query_vectors = [np.array(sentence_embedding("Owen").to_list()[0][0], dtype= 'float32')]

    print(query_vectors)

    res = client.search(
    collection_name=collection_name,     # target collection
    data=query_vectors,                # query vectors
    limit=3,                           # number of returned entities
    output_fields=['title']
    )

    print(res)

    #res = search_pipe('funny python demo')
    #DataCollection(res).show()
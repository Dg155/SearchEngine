from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops, DataCollection
import sys
import os
import pandas as pd

connections.connect(host='127.0.0.1', port='19530')

search_pipe = (pipe.input('query')
                    .map('query', 'vec', ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
                    .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
                    .flat_map('vec', ('id', 'score'), ops.ann_search.milvus_client(host='127.0.0.1', 
                                                                                   port='19530',
                                                                                   collection_name='search_article_in_medium'))  
                    .output('query', 'id', 'score')
               )


insert_pipe = (pipe.input('df')
                    .flat_map('df', 'data', lambda df: df.values.tolist())
                    .map('data', 'res', ops.ann_insert.milvus_client(host='127.0.0.1', 
                                                                        port='19530',
                                                                        collection_name='search_article_in_medium'))
                    .output('res')
    )

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),   
            FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            
    ]
    schema = CollectionSchema(fields=fields, description='search text')
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='title_vector', index_params=index_params)
    return collection

def insertData(Filepath):

   

    df = pd.read_csv(Filepath, "r") 
    insert_pipe(df)

    



if __name__ == "__main__":
    collection = create_milvus_collection('121Analyst', 768)

    collection.load()
    print(collection.num_entities)

    res = search_pipe('funny python demo')
    DataCollection(res).show()
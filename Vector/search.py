from pymilvus import DataType, MilvusClient
from towhee import AutoPipes, pipe, ops, DataCollection
import sys
import os
import pandas as pd
import numpy as np


collection_name = "Test"


if __name__ == "__main__":  

    client = MilvusClient(
    uri="http://localhost:19530"
    )
    sentence_embedding = AutoPipes.pipeline('sentence_embedding')

    while(1):
        print("Please input your query")
        query = input()

        print("Please input the amount of results you would like")
        count = int(input())

        query_vectors = [np.array(sentence_embedding(query).to_list()[0][0], dtype= 'float32')]

        res = client.search(
        collection_name=collection_name,     # target collection
        data=query_vectors,                # query vectors
        limit=count,                           # number of returned entities
        output_fields=['title', 'link']
        )

        print(f"----------Top {count} results----------")

        index = 0
        for response in res[0]:
            print(f"Response # {index} is {response['entity']['title']} at {response['entity']['link']}")
            print("-------------------------------------")    
            index += 1

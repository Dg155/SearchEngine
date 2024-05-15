from pymilvus import MilvusClient
from towhee import AutoPipes
import numpy as np
import  time


collection_name = "DEV"

if __name__ == "__main__":  
    client = MilvusClient(
    uri="https://in03-2893550dac2ce6c.api.gcp-us-west1.zillizcloud.com",
    token="69982138f98274926e691ee7f5c44a775f816ce0065d51fbb3fa95fbdb6be466c9f2d3c603354518c8cafcce30af0b3948049dcc"
    )
    sentence_embedding = AutoPipes.pipeline('sentence_embedding')

    while(1):
        print("Please input your query")
        query = input()

        print("Please input the amount of results you would like")
        count = int(input())

        start = time.time()
        query_vectors = [np.array(sentence_embedding(query).to_list()[0][0], dtype= 'float32')]
        end = time.time()
        print(f"Time to encode : {end - start}")
        
        start = time.time()
        res = client.search(
        collection_name=collection_name,     # target collection
        data=query_vectors,                # query vectors
        limit=count,                           # number of returned entities
        output_fields=['title', 'link']
        )

        end = time.time()
        print(f"Time to search : {end - start}")

        print(f"----------Top {count} results----------")

        index = 0
        for response in res[0]:
            print(f"Response # {index} is {response['entity']['title']} at {response['entity']['link']}")
            print("-------------------------------------")    
            index += 1


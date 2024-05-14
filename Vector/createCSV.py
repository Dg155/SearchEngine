import numpy
from towhee import pipe, ops, DataCollection
import csv
import os
import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from towhee import AutoPipes, AutoConfig

def CreateCSV():

    
    config = AutoConfig.load_config('sentence_embedding')
    config.model = 'sentence-t5-xxl'

    vectorPipe = AutoPipes.pipeline('sentence_embedding', config=config)



    with open('fileInfo.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["id", "title", "title_vector", "link", "content", "content_vector"]
        writer.writerow(field)
        
        fileList = GatherFiles("/content/drive/MyDrive/121Data/ANALYST")

        index = 0
        for file in fileList:
            print(f"Creating entry for {file}")
            # Read the files in
            fileInfo = ReadJSONFile(file)
            title, link, content = fileInfo

            # Create the vectors
            title_vector =  DataCollection(vectorPipe(title).get()).to_list()
            content_vector = DataCollection(vectorPipe(content).get()).to_list()
            
            writer.writerow([index, title, title_vector, link, content, content_vector])
            index += 1


def ReadJSONFile(filePath):
    with open(filePath, "r") as f:
        jsonData = json.load(f)
        soup = BeautifulSoup(jsonData["content"], "html.parser", from_encoding=jsonData["encoding"])
        title = soup.find('title').get_text() if soup.find('title') else None
        return title, jsonData["url"], soup.get_text()

def GatherFiles(folderPath):
    file_list = []
    for root, dirs, files in os.walk(folderPath):
        for file in files:
            # Only read json files
            if file.endswith(".json"):
                file_list.append(os.path.join(root, file))
    
    return file_list


if __name__ == "__main__":  
    CreateCSV()

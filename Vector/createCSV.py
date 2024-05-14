import numpy
from towhee import pipe, ops, DataCollection
import csv
import os
import json
from bs4 import BeautifulSoup
from towhee import AutoPipes, AutoConfig

def CreateCSV():

    sentence_embedding = AutoPipes.pipeline('sentence_embedding')

    with open('fileInfo.csv', 'w', newline='', encoding= 'utf-8') as file:
        writer = csv.writer(file)
        field = ["id", "title", "link", "content_vector"]
        writer.writerow(field)
        
        fileList = GatherFiles(r"C:\Users\kidro\OneDrive\Desktop\School\SearchEngine\ANALYST")

        index = 0
        for file in fileList:
            print(f"Creating entry for {file}")
            # Read the files in
            fileInfo = ReadJSONFile(file)
            title, link, content = fileInfo

            # Check for fields
            if(title == ""):
               title = "No title found"
            else:
                title = title.strip()
            content_vector =  sentence_embedding(content).to_list()[0][0]
            writer.writerow([index, title, link, content_vector])
            index += 1


def ReadJSONFile(filePath):
    with open(filePath, "r") as f:
        jsonData = json.load(f)
        soup = BeautifulSoup(jsonData["content"], "html.parser", from_encoding=jsonData["encoding"])
        title = soup.find('title').get_text() if soup.find('title') else ""
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

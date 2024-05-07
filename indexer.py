import os
import json
import nltk

def readJsonFiles(folder_path):
    json_set = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                    json_set.add(json.dumps(json_data))

        for dir in dirs:
            json_set.union(readJsonFiles(os.path.join(root, dir)))
    print(len(json_set))
    return json_set

def tokenizeJsonFiles(json_set):
    tokenized_set = set()
    for json_data in json_set:
        tokens = nltk.word_tokenize(json_data)
        tokenized_set.add(tuple(tokens))
    return tokenized_set
    

if __name__ == "__main__":

    currentPath = os.getcwd()
    analyst_folder = "\ANALYST"
    dev_folder = "\DEV"

    analyst_json_set = readJsonFiles(currentPath + analyst_folder)
    dev_json_set = readJsonFiles(currentPath + dev_folder)
    tokenizeJsonFiles(analyst_json_set)
    tokenizeJsonFiles(dev_json_set)
    

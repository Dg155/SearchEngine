import os
import json
import nltk
import shelve
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import pickle
from Posting import Posting

totalFiles = 100

def readJsonFiles(folderPath):

    jsonSet = set()
    for root, dirs, files in os.walk(folderPath):

        for file in files:
            # Only read json files
            if file.endswith(".json"):
                filePath = os.path.join(root, file)
                with open(filePath, "r") as f:

                    jsonData = json.load(f)
                    jsonSet.add(json.dumps(jsonData))

                    # Uncomment in case we want to limit our data set for time
                    # if len(jsonSet) == totalFiles:
                    #     return jsonSet

        print(f"Finished reading {root}")

        # Recurse through the directories to get all jsons
        for dir in dirs:
            jsonSet.union(readJsonFiles(os.path.join(root, dir)))

    return jsonSet

def parseDocumentIntoTokens(jsonFile):

    content = jsonFile["content"]
    # Check to ensure that content is html
    if not content.startswith("<!DOCTYPE html>") and not content.startswith("<html>"):
        return Counter()
    
    soup = BeautifulSoup(content, "html.parser")

    # BeautifulSoup shouuld be able to handle bad HTML, but just in case include some error handling
    if soup:
        try: # Try Catch for getting the text from the soup

            totalText = soup.get_text()

            if len(totalText) != 0:
                tokens = [string for string in nltk.word_tokenize(totalText) if len(string) > 1] # Remove single character tokens like 's' and ','
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                return Counter(tokens) # Transform into a dictionary of token strings and their frequency
            
        except Exception as e:
            print(e)
            return Counter()
        
    return Counter()
        

def buildIndex(jsonSet):
    # Function from lecture notes

    indexHashTable = defaultdict(list)
    index = 0
    documentsSkipped = 0

    for jsonFile in jsonSet:

        jsonFile = json.loads(jsonFile)
        index += 1
        tokens = parseDocumentIntoTokens(jsonFile)
        # It says remove duplicates but we dont have to because we are counting frequency?

        if len(tokens) != 0:
            for token, count in tokens.items():
                indexHashTable[token].append(Posting(index, count, jsonFile["url"]))
                #print("Indexed document ", jsonFile["url"])
        else:
            print("No tokens found in document ", jsonFile["url"])
            documentsSkipped += 1
            index -= 1
    
    # Save the index data to shelve
    with shelve.open("indexTable.shelve") as indexTable:
        if len(indexTable) != 0:
            del indexTable["indexTable"]
        indexTable["indexTable"] = indexHashTable
        indexTable.sync()
    
    # SortAndWriteToDisk(indexHashTable, name)
    with open("indexHashTable.pickle", "ab") as name:
        pickle.dump(indexHashTable, name)
    
    # Record the number of indexed documents, number of skipped documents, unique tokens, top tokens, and the total size of the index
    # (Might be worth moving this outside of this function)
    with shelve.open("indexedDocuments.shelve") as indexedDocuments:
        indexedDocuments["indexedDocumesnts"] = index
        indexedDocuments.sync()
    with shelve.open("skippedDocuments.shelve") as skippedDocuments:
        skippedDocuments["skippedDocuments"] = documentsSkipped
        skippedDocuments.sync()
    with shelve.open("uniqueTokens.shelve") as uniqueTokens:
        uniqueTokens["uniqueTokens"] = len(indexHashTable)
        uniqueTokens.sync()
    with shelve.open("topTokens.shelve") as topTokens:
        topTokens["topTokens"] = {key:len(value) for (key,value) in sorted(indexHashTable.items(), key=lambda x: len(x[1]), reverse=True)[:50]}
        topTokens.sync()
    with shelve.open("totalSize.shelve") as totalSize:
        totalSize["kilobytes"] = "{:.2f} KB".format(os.path.getsize('indexHashTable.pickle') / 1024)
        totalSize.sync()
    return indexHashTable

if __name__ == "__main__":

    currentPath = os.getcwd()
    analyst_folder = "\ANALYST"
    dev_folder = "\DEV"

    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = nltk.stem.WordNetLemmatizer()


    with shelve.open("jsonSet.shelve") as jsonSet:

        if len(jsonSet) == 0:
            jsonSet["AnalystJson"] = (readJsonFiles(currentPath + analyst_folder))
            jsonSet.sync()
            print("Analyst data set loaded")
            jsonSet["DevJson"] = (readJsonFiles(currentPath + dev_folder))
            print("Dev data set loaded")
            jsonSet.sync()
            

        else: 
            print("Data already exists in the shelve")

        buildIndex(jsonSet["AnalystJson"])
        jsonSet.sync()
        buildIndex(jsonSet["DevJson"])
        jsonSet.sync()


    

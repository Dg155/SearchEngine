import os
import json
import nltk
import shelve
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import pickle
from Posting import Posting
import re
import threading

lock = threading.Lock()
def parseDocumentIntoTokens(jsonFile):

    content = jsonFile["content"]
    if not content.startswith("<!DOCTYPE html>") and not content.startswith("<html>"):
        return Counter()
    soup = BeautifulSoup(content, "html.parser")
    # BeautifulSoup shouuld be able to handle bad HTML, but just in case include some error handling
    if soup:
        try: # Try Catch for getting the text from the soup

            totalText = soup.get_text()

            if len(totalText) != 0:
                tokens = [string for string in nltk.word_tokenize(totalText) if len(string) > 1] 
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                return Counter(tokens)
        except Exception as e:
            print(e)
            return Counter()
        
    return Counter()


def GatherFiles(folderPath):
    file_list = []
    dirs = []
    for root, dirs, files in os.walk(folderPath):
        for file in files:
            # Only read json files
            if file.endswith(".json"):
                file_list.append(os.path.join(root, file))
        
    
    for dir in dirs:
        file_list.append(GatherFiles(dir))

    return file_list


def ReadJSONFile(filePath):

    with open(filePath, "r") as f:
        jsonData = json.load(f)
        return (jsonData, jsonData["url"])

def BuildPosting(currentPostings : dict, tokenFrequencies : Counter, url, id):
    
    for token, count in tokenFrequencies.items():
        if token in currentPostings.keys():
            currentPostings[token].append(pickle.dumps(Posting(id, count, url)))
        else:
            currentPostings[token] = [pickle.dumps(Posting(id, count, url))]  # Initialize as a list


# Found this online lol
def chunk_list(lst, chunk_size):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def process_batch(batch, id_offset):
    global savedPostings
    postings = defaultdict(list)
    for file in batch:
        jsonFile, url = ReadJSONFile(file)
        tokens = parseDocumentIntoTokens(jsonFile)
        if len(tokens) > 0:
            BuildPosting(postings, tokens, url, id_offset)
            id_offset += 1
            print(f"adding file : {id_offset}")
    with lock:
        for key, value in postings.items():
            if key in savedPostings:
                savedPostings[key] += value
            else:
                savedPostings[key] = value
        savedPostings.sync()

if __name__ == "__main__":

    currentPath = os.getcwd()

    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    currentPath = os.getcwd()
    folderpath = "\DEV"
    file_list = GatherFiles(currentPath+folderpath)

    #table with assorted numbers pertaining to your index. It should have,
    #at least the number of documents, the number of [unique] tokens, and the
    #total size (in KB) of your index on disk

    # arbitrary number
    chunks = chunk_list(file_list, len(file_list))

    # path to current postings
    savedPostings = shelve.open("postings.shelve")
    id_offset = 0
    
    # Divide file list into chunks
    chunks = chunk_list(file_list, int(len(file_list)/25))

    # Create and start threads
    threads = []
    previousOffset = 0
    for chunk in chunks:
        thread = threading.Thread(target=process_batch, args=(chunk, id_offset))
        thread.start()
        threads.append(thread)
        id_offset += len(chunk)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    

    print("{:.2f} KB".format(os.path.getsize("postings.shelve") / 1024))
    print(f"Unique count : {len(savedPostings)}")
    print(f"Count = {id_offset}")

    savedPostings.close()




        

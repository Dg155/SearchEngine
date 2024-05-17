import os
import json
import nltk
import shelve
from collections import Counter
from bs4 import BeautifulSoup
from Posting import Posting
import psutil
import re
from nltk.stem import PorterStemmer



simHashQueueLength = 100
simHashThreshold = 0.85
batchSize = 15000
readingMemoryLimit = 1000
indexingMemoryLimit = 500
ps = PorterStemmer()

def intOrFloat(s):
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)$'
    return bool(re.fullmatch(pattern, s))

def readandIndexJsonFiles(folderPath):

    invertedIndexID = 0
    documentsSkipped = 0
    uniqueWords = set()
    jsonSet = set()

    for root, dirs, files in os.walk(folderPath):

        for file in files:
            # Only read json files
            if file.endswith(".json"):
                filePath = os.path.join(root, file)
                jsonSet.add(filePath)

                memory = psutil.virtual_memory() # Check memory usage

                if memory.available / (1024 ** 2) < readingMemoryLimit:
                    print(f"Memory limit reached; current memory: {memory.available / (1024 ** 2)} MB; indexing {len(jsonSet)} documents")
                    invertedIndexID, documentsSkipped, uniqueWords = buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords)
                    jsonSet.clear()

                if len(jsonSet) > batchSize:
                    print(f"Batch size reached, indexing {batchSize} documents")
                    invertedIndexID, documentsSkipped, uniqueWords = buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords)
                    jsonSet.clear()
    
    if len(jsonSet) > 0:
        print(f"Indexing remaining {len(jsonSet)} documents")
        invertedIndexID, documentsSkipped, uniqueWords = buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords)
        jsonSet.clear()

        print(f"Finished reading {root}")
    
    print(f"Total documents indexed: {invertedIndexID}. Total documents skipped: {documentsSkipped}. Total unique tokens: {len(uniqueWords)}")

    return invertedIndexID, documentsSkipped, uniqueWords

def parseDocumentIntoTokens(jsonFile):

    content = jsonFile["content"]
    
    soup = BeautifulSoup(content, "html.parser")

    # BeautifulSoup shouuld be able to handle bad HTML, but just in case include some error handling
    if soup:
        try: # Try Catch for getting the text from the soup

            title = soup.find('title').text if soup.find('title') else None
            totalText = soup.get_text()

            if not checkDuplicate(soup):
                return "", Counter()

            if len(totalText) != 0:
                tokens = [string for string in nltk.word_tokenize(totalText) if len(string) > 1] # Remove single character tokens like 's' and ','
                tokens = [ps.stem(token) for token in tokens if not intOrFloat(token)]
                return title, Counter(tokens) # Transform into a dictionary of token strings and their frequency
            
        except Exception as e:
            print(e)
            return "", Counter()
        
    return "", Counter()
        

def buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords):

    indexedHashtable = dict()
    mapOfUrls = dict()

    for filePath in jsonSet:

        with open(filePath, "r") as f:

            jsonData = json.load(f)
            title, tokens = parseDocumentIntoTokens(jsonData)

            if len(tokens) != 0:

                invertedIndexID += 1

                for token, count in tokens.items():
                    # Add to unique words data
                    uniqueWords.add(token)
                    # Add posting to inverted index
                    if token not in indexedHashtable:
                        indexedHashtable[token] = [Posting(invertedIndexID, count)]
                    else:
                        indexedHashtable[token].append(Posting(invertedIndexID, count))
                
                # Add to urlMap
                title = title if title else "Title Not Found"
                mapOfUrls[str(invertedIndexID)] = [title, jsonData["url"]]
                #print("Indexed document ", title)

                # Memory Checking
                memory = psutil.virtual_memory() # Check memory usage
                if memory.available / (1024 ** 2) < indexingMemoryLimit: # Ensure we respect memory limits so we dont error
                    print(f"Memory limit reached; current memory: {memory.available / (1024 ** 2)} MB; indexing {invertedIndexID} documents")
                    with shelve.open("DevInvertedIndex.shelve") as invertedIndex:

                        for key, value in indexedHashtable.items():
                            if key in invertedIndex:
                                invertedIndex[key] += value
                            else:
                                invertedIndex[key] = value

                        indexedHashtable.clear()

                        invertedIndex.sync()

            else:
                #print("No tokens found in document ", jsonData["url"])
                documentsSkipped += 1
    
    print(f"Indexed {len(jsonSet)} documents. Writing to disk.")

    with shelve.open("DevInvertedIndex.shelve") as invertedIndex:

        for key, value in indexedHashtable.items():
            if key in invertedIndex:
                invertedIndex[key] += value
            else:
                invertedIndex[key] = value

        invertedIndex.sync()

    with shelve.open("DevUrlMap.shelve") as urlMap:
        urlMap.update(mapOfUrls)
        urlMap.sync()

    return invertedIndexID, documentsSkipped, uniqueWords



def checkDuplicate(soup):
    # For exact duplicates, use CRC to hash the page and compare to all previously visited pages.
    totalText = soup.get_text()
    crcHash = cyclic_redundancy_check(totalText)

    # Reserve dict for this thread
    with shelve.open("hashOfPages.shelve") as hashOfPages:
        # need to be str so the key lookup works
        if str(crcHash) in hashOfPages:
            #print("Duplicate page found")
            return False
        else:
            hashOfPages[str(crcHash)] = True

    # Check for near duplicates with simhashes
    with shelve.open("simHashSetLock.shelve") as simHashSetLock:

        if "Queue" not in simHashSetLock:
            simHashSetLock["Queue"] = []

        sim_hash = simHash(totalText)

        # Maintain a reasonable queue of links to compare to
        while len(simHashSetLock["Queue"]) > simHashQueueLength:
                simHashSetLock["Queue"].pop(0)

        
        for sim in simHashSetLock["Queue"]:
            if areSimilarSimHashes(sim_hash, sim, simHashThreshold):
                #print("Near duplicate page found")
                return False

        # Set the existance of the hash in the shelve
        simHashSetLock["Queue"].append(sim_hash)

    return True

def cyclic_redundancy_check(pageData):
    
    crcHash = 0xFFFF

    # Convert page data into bytes and iterate
    for byte in pageData.encode():

        crcHash ^= byte # bitwise XOR

        for _ in range(8): # 8 bits in a byte
            if crcHash & 0x0001: # Check LSB 
                crcHash = (crcHash >> 1) ^ 0xA001 # Polynomial divison process
            else:
                crcHash >>= 1 # Shift right by 1 bit to discard LSB

    # Return compliment of hash
    return crcHash ^ 0xFFFF

def simHash(pageData):
    # Seperate into words with weights
    weightedWords = Counter([ps.stem(string) for string in nltk.word_tokenize(pageData) if len(string) > 1])

    # Get 8-bit hash values for every unique word
    hashValues = {word: bit_hash(word) for word in weightedWords}

    # Calculate the Vector V by summing weights
    simhash_value = [0] * 8

    for word, count in weightedWords.items():
        wordHash = hashValues[word]

        for i in range(8):
            # Offset hash digit by index of range, and multiply by 1 to get LSB
            bit = (wordHash >> i) & 1
            simhash_value[i] += (1 if bit else -1) * count
    
    # Convert into fingerprint
    simhash_fingerprint = 0
    for i in range(8):
        if simhash_value[i] > 0:
            simhash_fingerprint |= 1 << i
    
    return simhash_fingerprint

def bit_hash(word):

    # Function to generate an 8 bit hash for a word
    hash = 0

    for character in word:
        # Add ASCII value to total hash
        hash += ord(character)
    
    # Ensure hash value is 8 bits
    return hash % 256

def areSimilarSimHashes(firstSimHash, secondSimHash, threshold):
    # Return true if two hashes are similar, else false

    # Get number of different bits by XOR the two hashes, and count the occurances of 0 (similarities)
    similarBits = bin(firstSimHash ^ secondSimHash).count('0')
    # Calculate the similarity ratio
    similarity = similarBits / 8

    #if similarity >= threshold:
    #    get_logger("CRAWLER").warning(f"simHash similarity: {similarity} already visited")

    return similarity >= threshold

if __name__ == "__main__":

    currentPath = os.getcwd()
    analyst_folder = "\ANALYST"
    dev_folder = "\DEV"
    
    # lemmatizer = nltk.stem.WordNetLemmatizer()

    # invertedIndexID, documentsSkipped, uniqueWords = readandIndexJsonFiles(currentPath + analyst_folder)
    # print("Analyst data set loaded and indexed")

    # with shelve.open("analystInfo.shelve") as analystInfo:

    #     if len(analystInfo) == 0:
    #         invertedIndexID, documentsSkipped, uniqueWords = readandIndexJsonFiles(currentPath + analyst_folder)
    #         print("Analyst data set loaded and indexed")
            
    #         analystInfo["indexedDocumesnts"] = invertedIndexID
    #         analystInfo["skippedDocuments"] = documentsSkipped
    #         analystInfo["uniqueTokens"] = len(uniqueWords)
    #         analystInfo["kilobytes"] = "{:.2f} KB".format(os.path.getsize('analystInvertedIndex.shelve.bak') / 1024)
    #         analystInfo.sync()

    with shelve.open("devInfo.shelve") as devInfo:

        if len(devInfo) == 0:
            invertedIndexID, documentsSkipped, uniqueWords = readandIndexJsonFiles(currentPath + dev_folder)
            print("Analyst data set loaded and indexed")
            
            devInfo["indexedDocumesnts"] = invertedIndexID
            devInfo["skippedDocuments"] = documentsSkipped
            devInfo["uniqueTokens"] = len(uniqueWords)
            devInfo["kilobytes"] = "{:.2f} KB".format(os.path.getsize('analystInvertedIndex.shelve.bak') / 1024)
            devInfo.sync()


    

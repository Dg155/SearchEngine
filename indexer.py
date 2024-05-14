import os
import json
import nltk
import shelve
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import pickle
from Posting import Posting

simHashQueueLength = 100
simHashThreshold = 0.65
batchSize = 1000
exploredRoots = set()

def readandIndexJsonFiles(folderPath):

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
    
    with shelve.open("jsonSet.shelve") as jsonSetShelve:
        jsonSetShelve["jsonSet"] = jsonSet
        print("Json set saved to shelve")
        jsonSetShelve.sync()
        
    buildIndex(list(jsonSet))

    return jsonSet

def parseDocumentIntoTokens(jsonFile):

    content = jsonFile["content"]
    
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
    uniqueWords = set()

    for jsonFile in jsonSet:

        jsonFile = json.loads(jsonFile)
        index += 1
        tokens = parseDocumentIntoTokens(jsonFile)
        # It says remove duplicates but we dont have to because we are counting frequency?

        if len(tokens.items()) != 0:
            for token, count in tokens.items():
                indexHashTable[token].append(Posting(index, count, jsonFile["url"]))
                #print("Indexed document ", jsonFile["url"])
        else:
            batchList = documentList
            documentList = []
        
        for jsonFile in batchList:

            jsonFile = json.loads(jsonFile)
            index += 1
            tokens = parseDocumentIntoTokens(jsonFile)
            # It says remove duplicates but we dont have to because we are counting frequency?

            if len(tokens) != 0:
                for token, count in tokens.items():
                    uniqueWords.add(token)
                    indexHashTable[token].append(Posting(index, count, jsonFile["url"]))
                    #print("Indexed document ", jsonFile["url"])
            else:
                print("No tokens found in document ", jsonFile["url"])
                documentsSkipped += 1
                index -= 1
    
        # SortAndWriteToDisk(indexHashTable, name)
        with open("indexHashTable.pickle", "ab") as name:
            pickle.dump(indexHashTable, name)
            indexHashTable.clear()
            print(f"Batch size {len(batchList)} indexed")

    
    # Record the number of indexed documents, number of skipped documents, unique tokens, top tokens, and the total size of the index
    # (Might be worth moving this outside of this function)
    with shelve.open("indexedDocuments.shelve") as indexedDocuments:
        indexedDocuments["indexedDocumesnts"] = index
        indexedDocuments.sync()
    with shelve.open("skippedDocuments.shelve") as skippedDocuments:
        skippedDocuments["skippedDocuments"] = documentsSkipped
        skippedDocuments.sync()
    with shelve.open("uniqueTokens.shelve") as uniqueTokens:
        uniqueTokens["uniqueTokens"] = uniqueWords
        uniqueTokens.sync()
    with shelve.open("totalSize.shelve") as totalSize:
        totalSize["kilobytes"] = "{:.2f} KB".format(os.path.getsize('indexHashTable.pickle') / 1024)
        totalSize.sync()


def checkDuplicate(soup):
    # For exact duplicates, use CRC to hash the page and compare to all previously visited pages.
    totalText = soup.get_text()
    crcHash = cyclic_redundancy_check(totalText)

    # Reserve dict for this thread
    with shelve.open("hashOfPages.shelve") as hashOfPages:
        # need to be str so the key lookup works
        if str(crcHash) in hashOfPages:
            #crawler.logger.warning(f"hash: {crcHash}  for url {resp.url} already visited")
            return False
        else:
            hashOfPages[str(crcHash)] = True

    # Check for near duplicates with simhashes
    with shelve.open("simHashSetLock.shelve") as simHashSetLock:
        sim_hash = simHash(totalText)

        # Maintain a reasonable queue of links to compare to
        while len(simHashSetLock["Queue"]) > simHashQueueLength:
                simHashSetLock["Queue"].pop(0)

        
        for sim in simHashSetLock["Queue"]:
            if areSimilarSimHashes(sim_hash, sim, simHashThreshold):
                #crawler.logger.warning(f"high similarity on {resp.url}")
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
    weightedWords = Counter([lemmatizer.lemmatize(string) for string in nltk.word_tokenize(pageData) if len(string) > 1])

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

    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = nltk.stem.WordNetLemmatizer()


    #readandIndexJsonFiles(currentPath + analyst_folder)
    #print("Analyst data set loaded and indexed")
    with shelve.open("jsonSet.shelve") as jsonSetShelve:
        if len(jsonSetShelve) != 0:
            print("Loading dev data set from shelve")
            jsonSet = jsonSetShelve["jsonSet"]
            buildIndex(jsonSet)
            print("Dev data set loaded and indexed")
        else:
            readandIndexJsonFiles(currentPath + dev_folder)
            print("Dev data set loaded and indexed")


    

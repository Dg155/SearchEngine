import os
import json
import nltk
import shelve
from collections import Counter
from bs4 import BeautifulSoup
from Posting import Posting
import psutil
import re
import math
from nltk.stem import PorterStemmer
from transformers import pipeline
import sys

simHashQueueLength = 100
simHashThreshold = 0.85
readingMemoryLimit = 1000
indexingMemoryLimit = 500
ps = PorterStemmer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(content):
    return ''
    try:
        summary = summarizer(content, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return content[:150] + '...'

def isValidToken(s):
    pattern = r'[a-zA-Z]'
    scientificNotationPattern = r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$'
    hasLetter = re.search(pattern, s) is not None
    singleSlash = s.count('/') <= 1
    noTilda = s.count('~') == 0
    notScientificNotation = not re.match(scientificNotationPattern, s)
    return hasLetter and singleSlash and noTilda and notScientificNotation

def readandIndexJsonFiles(folderPath):

    invertedIndexID = 0
    documentsSkipped = 0
    uniqueWords = set()
    jsonSet = set()
    fileNumber = 1

    for root, dirs, files in os.walk(folderPath):

        for file in files:
            # Only read json files
            if file.endswith(".json"):
                filePath = os.path.join(root, file)
                print(f"Reading {filePath}")
                jsonSet.add(filePath)

                memory = psutil.virtual_memory() # Check memory usage

                # If virtual memory limit is reached, begin indexing
                if memory.available / (1024 ** 2) < readingMemoryLimit:
                    print(f"Memory limit reached; current memory: {memory.available / (1024 ** 2)} MB; indexing {len(jsonSet)} documents")
                    invertedIndexID, documentsSkipped, uniqueWords, fileNumber = buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords, fileNumber)
                    jsonSet.clear()

                # If batch size is reached, begin indexing
                if len(jsonSet) > batchSize:
                    print(f"Batch size reached, indexing {batchSize} documents")
                    invertedIndexID, documentsSkipped, uniqueWords, fileNumber = buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords, fileNumber)
                    jsonSet.clear()

        print(f"Finished reading root: {root}")
    
    # Index any remaining documents in the json set
    if len(jsonSet) > 0:
        print(f"Indexing remaining {len(jsonSet)} documents")
        invertedIndexID, documentsSkipped, uniqueWords, fileNumber = buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords, fileNumber)
        jsonSet.clear()
    
    print(f"Total documents indexed: {invertedIndexID}. Total documents skipped: {documentsSkipped}. Total unique tokens: {len(uniqueWords)}")

    return invertedIndexID, documentsSkipped, uniqueWords

def parseDocumentIntoTokens(jsonFile):

    content = jsonFile["content"]
    
    soup = BeautifulSoup(content, "html.parser", from_encoding=jsonFile["encoding"] if "encoding" in jsonFile else "utf-8")

    # BeautifulSoup shouuld be able to handle bad HTML, but just in case include some error handling
    if soup:
        try: # Try Catch for getting the text from the soup

            title = soup.find('title').text if soup.find('title') else None
            totalText = soup.get_text()

            if not checkDuplicate(soup):
                return "", Counter(), ""

            if len(totalText) != 0:
                summary = summarize_text(totalText)
                tokens = [ps.stem(string) for string in nltk.word_tokenize(totalText) if len(string) > 1 and len(string) < 46 and isValidToken(string)] # Remove single character tokens like 's' and ','
                return title, Counter(tokens), summary # Transform into a dictionary of token strings and their frequency
            
        except Exception as e:
            print(e)
            return "", Counter(), ""
        
    return "", Counter(), ""
        

def buildIndex(jsonSet, invertedIndexID, documentsSkipped, uniqueWords, fileNumber):

    indexedHashtable = dict()
    mapOfUrls = dict()

    for filePath in jsonSet:

        with open(filePath, "r") as f:

            jsonData = json.load(f)
            title, tokens, summary = parseDocumentIntoTokens(jsonData)

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
                mapOfUrls[str(invertedIndexID)] = [title, jsonData["url"], summary]
                print(f"Indexed document #{invertedIndexID+documentsSkipped}: {title}")

                # Memory Checking
                memory = psutil.virtual_memory() # Check memory usage
                if memory.available / (1024 ** 2) < indexingMemoryLimit: # Ensure we respect memory limits so we dont error

                    print(f"Memory limit reached; current memory: {memory.available / (1024 ** 2)} MB; indexing {invertedIndexID} documents")

                    # Write the remaining index to disk
                    fileName = f"InvertedIndex_{fileNumber}.txt"
                    with open(fileName, "w") as file:
                        for token in sorted(indexedHashtable.keys()):
                            try: # Try catch to handle any encoding errors
                                postings = indexedHashtable[token]
                                postingString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in postings)
                                file.write(f"{token}~{postingString}\n")
                            except Exception as e:
                                continue
                        totalTextFiles.append(fileName)
                        fileNumber += 1
                        indexedHashtable.clear()

            else:
                print("No tokens found in document #", invertedIndexID+documentsSkipped)
                documentsSkipped += 1
    
    print(f"Indexed {len(jsonSet)} documents. Writing to disk.")

    # Write the remaining index to disk
    fileName = f"InvertedIndex_{fileNumber}.txt"
    with open(fileName, "w") as file:
        for token in sorted(indexedHashtable.keys()):
            try: # Try catch to handle any encoding errors
                postings = indexedHashtable[token]
                postingString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in postings)
                file.write(f"{token}~{postingString}\n")
            except Exception as e:
                continue
        totalTextFiles.append(fileName)
        fileNumber += 1
        indexedHashtable.clear()

    print(f"Finished writing to disk, updating url map.")

    # Write the url map to disk
    with open("URLMap.json", "w") as f:
        json.dump(mapOfUrls, f)

    return invertedIndexID, documentsSkipped, uniqueWords, fileNumber

def ParseLineToKeyPostingPair(line):
    # Break key and list of postings into separate variables
    key, postingsString = line.strip().split('~')

    documentFrequency = len(postingsString.split(',')) # Number of documents containing the term

    postings = []
    # Iterate through each posting in the list
    for posting in postingsString.split(','):
        posting = posting.strip('[]').split(';') # Remove brackets and split into values
        docID = int(posting[0])
        count = int(posting[1])
        termFreq = 1 + math.log(count, 10) if count > 0 else 0
        inverseDocFreq = math.log((invertedIndexID / documentFrequency), 10) # Log base 10 of 1 + 1
        postings.append(Posting(docID, count, tf=termFreq, idf=inverseDocFreq))
    return key, postings

def combinePostings(postingList1, postingList2):
    # Combine two posting lists into one
    combinedPostings = []
    i = 0
    j = 0

    while i < len(postingList1) and j < len(postingList2):
        if postingList1[i].docID == postingList2[j].docID:
            combinedPostings.append(Posting(postingList1[i].docID, postingList1[i].count + postingList2[j].count))
            i += 1
            j += 1
        elif postingList1[i].docID < postingList2[j].docID:
            combinedPostings.append(postingList1[i])
            i += 1
        else:
            combinedPostings.append(postingList2[j])
            j += 1
    
    # Add remaining postings from list1
    while i < len(postingList1):
        combinedPostings.append(postingList1[i])
        i += 1
    
    # Add remaining postings from list2
    while j < len(postingList2):
        combinedPostings.append(postingList2[j])
        j += 1
    
    return combinedPostings

def combineFiles(file1, file2, output_file):

    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as outf:
        # Read first line from each file
        line1 = f1.readline().strip()
        line2 = f2.readline().strip()
        
        while line1 and line2: # While there are lines in both files

            # Parse lines into key and postings
            key1, postingList1 = ParseLineToKeyPostingPair(line1)
            key2, postingList2 = ParseLineToKeyPostingPair(line2)
            
            if key1 == key2:
                combinedPostings = combinePostings(postingList1, postingList2)
                postingsString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in combinedPostings)
                outf.write(f"{key1}~{postingsString}\n")
                line1 = f1.readline().strip()
                line2 = f2.readline().strip()
            elif key1 < key2:
                postingsString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in postingList1)
                outf.write(f"{key1}~{postingsString}\n")
                line1 = f1.readline().strip()
            else:
                postingsString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in postingList2)
                outf.write(f"{key2}~{postingsString}\n")
                line2 = f2.readline().strip()
        
        # Write remaining lines from file1
        while line1:
            key1, postingList1 = ParseLineToKeyPostingPair(line1)
            postingsString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in postingList1)
            outf.write(f"{key1}~{postingsString}\n")
            line1 = f1.readline().strip()
        
        # Write remaining lines from file2
        while line2:
            key2, postingList2 = ParseLineToKeyPostingPair(line2)
            postingsString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in postingList2)
            outf.write(f"{key2}~{postingsString}\n")
            line2 = f2.readline().strip()

    return output_file



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
    weightedWords = Counter([ps.stem(string) for string in nltk.word_tokenize(pageData) if len(string) > 1 and len(string) < 46 and isValidToken(string)])

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

def generateSingleFileIndex(inputFile, outputFile):
        
    with open(inputFile, 'r') as f, open(outputFile, 'w') as outf:
        line = f.readline().strip()
        while line:
            key, postingList = ParseLineToKeyPostingPair(line)
            postingsString = ','.join(f'[{p.docID};{p.freq};{p.tf};{p.idf}]' for p in postingList)
            outf.write(f"{key}~{postingsString}\n")
            line = f.readline().strip()

def generateIndexOfIndex(indexFile):

    indexOfIndex = dict()
    currentIndex = 0

    with open(indexFile, 'r') as f:
        line = f.readline().strip()
        while line:
            key, _ = ParseLineToKeyPostingPair(line)
            indexOfIndex[key] = currentIndex
            currentIndex = f.tell()
            line = f.readline().strip()
    
    with open("indexOfIndex.json", "w") as f:
        json.dump(indexOfIndex, f)

if __name__ == "__main__":

    currentPath = os.getcwd()
    analyst_folder = "\ANALYST"
    dev_folder = "\DEV"

    totalTextFiles = []

    if len(sys.argv) > 2:
        argument = sys.argv[1]
        batchSize = int(sys.argv[2])
        if argument == "ANA" or argument == "DEV":
            currentPath = currentPath + analyst_folder if argument == "ANA" else currentPath + dev_folder
            with shelve.open(f"{argument}Info.shelve") as dataInfo:

                if len(dataInfo) == 0:
                    
                    print(f"Indexing {argument} data set with batch size: {batchSize}")
                    invertedIndexID, documentsSkipped, uniqueWords = readandIndexJsonFiles(currentPath)
                    print(f"{argument} data set loaded and indexed")

                    print("Combining Indexes")

                    if len(totalTextFiles) == 1:
                        print("Only one index file, no need to combine")
                        generateSingleFileIndex(totalTextFiles[0], "FinalCombined.txt")
                        os.remove(totalTextFiles[0])
                    else:
                        combinedIndex = totalTextFiles.pop(0)
                        i = 0
                        for index in totalTextFiles:
                            print(f"Combining {combinedIndex} and {index}")
                            if (i == len(totalTextFiles) - 1):
                                combinedIndex = combineFiles(combinedIndex, index, f"FinalCombined.txt")
                            else:
                                combinedIndex = combineFiles(combinedIndex, index, f"CombinedIndex_{i}.txt")
                            os.remove(index)
                            i += 1

                    print(f"Combined files, writing indexOfIndex")
                    # create the index of index hashmap and write it to json
                    generateIndexOfIndex("FinalCombined.txt")

                    print("Finished combining indexes")

                    
                    dataInfo["indexedDocumesnts"] = invertedIndexID
                    dataInfo["skippedDocuments"] = documentsSkipped
                    dataInfo["uniqueTokens"] = len(uniqueWords)
                    dataInfo["kilobytes"] = "{:.2f} KB".format(os.path.getsize('FinalCombined.txt') / 1024)
                    dataInfo.sync()

                    # Clean up data used for duplicate and near duplicate pages
                    os.remove("simHashSetLock.shelve.bak")
                    os.remove("simHashSetLock.shelve.dat")
                    os.remove("simHashSetLock.shelve.dir")
                    os.remove("hashOfPages.shelve.bak")
                    os.remove("hashOfPages.shelve.dat")
                    os.remove("hashOfPages.shelve.dir")
        else:
            print("Invalid argument. Please provide ANA or DEV.")
    else:
        print("Not enough arguments provided. Please provide ANA or DEV as well as Batch Size (int).")
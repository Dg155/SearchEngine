import shelve
import  time
import json
from nltk.stem import PorterStemmer
from Posting import Posting

def intersectPostings(posting1, posting2):
    # Use a double pointer method to merge two postings together by there intersection
    merged = []
    i, j = 0, 0

    while i < len(posting1) and j < len(posting2):

        if posting1[i].docID == posting2[j].docID:
            
            newPosting = Posting(posting1[i].docID, posting1[i].freq + posting2[j].freq, tf=posting1[i].tf + posting2[i].tf, idf=posting1[i].idf + posting2[i].idf)
            newPosting.tfidf = posting1[i].tfidf + posting2[i].tfidf
            merged.append(newPosting)
            i += 1
            j += 1

        elif posting1[i].docID < posting2[j].docID:

            i += 1

        else:

            j += 1

    return merged

def mergePostingLists(totalPostings):

    if not totalPostings:
        return []
    
    # Sort the posting lists by their length to optimize the intersection process
    totalPostings.sort(key=len)
    
    # Intersect two postings lists at a time
    finalPosting = totalPostings[0]

    for i in range(1, len(totalPostings)):

        finalPosting = intersectPostings(finalPosting, totalPostings[i])

        if not finalPosting: # Early exit if there are no common docIDs
            break
    
    return finalPosting

def ParseLineToKeyPostingPair(line):
    # Break key and list of postings into separate variables
    key, postingsString = line.strip().split('~')

    postings = []
    # Iterate through each posting in the list
    for posting in postingsString.split(','):
        posting = posting.strip('[]').split(';') # Remove brackets and split into values
        docID = int(posting[0])
        count = int(posting[1])
        termFreq = float(posting[2])
        inverseDocFreq = float(posting[3])
        postings.append(Posting(docID, count, tf=termFreq, idf=inverseDocFreq))
    return key, postings

if __name__ == "__main__":  

    ps = PorterStemmer()

    print("----------Welcome to the search engine----------")

    # Load index of index from json file
    with open("indexOfIndex.json", "r") as f:
        indexMap = json.load(f)
    # Load URL map from json file
    with open("URLMap.json", "r") as f:
        urlMap = json.load(f)

    while(1):
        print("Please input your query:")
        query = input()

        print("Please input the amount of results you would like:")
        count = int(input())
        
        start = time.time()

        # Find all query results

        totalQueries = [ps.stem(quer) for quer in query.split()]

        totalPostings = []
        
        for query in totalQueries:
            if query in indexMap:
                seekPosition = indexMap[query]
                with open("FinalCombined.txt", "r") as indexFile:
                    indexFile.seek(seekPosition)
                    line = indexFile.readline().strip()
                    key, postings = ParseLineToKeyPostingPair(line)
                    totalPostings.append(postings)
            else:
                totalPostings.append([])

        finalPostings = mergePostingLists(totalPostings) if len(totalPostings) > 1 else totalPostings[0] # Merge posting lists if necessary

        finalPostings.sort(key=lambda x: (x.tfidf), reverse=True) # Basic ranking by tf-idf

        if not finalPostings:
            print("----------No results found----------")
            continue

        end = time.time()
        print(f"Time to search : {end - start}")

        print(f"----------Top results----------")

        for i in range(count):
            if i >= len(finalPostings):
                print("----------No more results found----------")
                break
            urlInfo = urlMap[str(finalPostings[i].docID)]
            print(f"#{i+1}: {urlInfo[0]} ({urlInfo[1]})")
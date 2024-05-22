import shelve
import  time
from nltk.stem import PorterStemmer
from Posting import Posting

def intersectPostings(posting1, posting2):
    # Use a double pointer method to merge two postings together by there intersection
    merged = []
    i, j = 0, 0

    while i < len(posting1) and j < len(posting2):

        if posting1[i].docID == posting2[j].docID:
            
            merged.append(Posting(posting1[i].docID, posting1[i].freqCount + posting2[j].freqCount))
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

if __name__ == "__main__":  

    ps = PorterStemmer()

    while(1):
        print("Please input your query:")
        query = input()

        print("Please input the amount of results you would like:")
        count = int(input())
        
        start = time.time()

        # Find all query results

        totalQueries = [ps.stem(quer) for quer in query.split()]

        totalPostings = []
        
        with shelve.open("AnalystInvertedIndex.shelve") as invertedIndex:
            totalPostings = [invertedIndex[query] if query in invertedIndex else [] for query in totalQueries]

        # for postList in totalPostings:
        #     for post in postList:
        #         print(post.docID, end=" ")
        #     print()

        finalPostings = mergePostingLists(totalPostings) if len(totalPostings) > 1 else totalPostings[0] # Merge posting lists if necessary

        finalPostings.sort(key=lambda x: x.freqCount, reverse=True) # Basic ranking by frequency count

        if not finalPostings:
            print("----------No results found----------")
            continue

        end = time.time()
        print(f"Time to search : {end - start}")

        print(f"----------Top results----------")

        with shelve.open("AnalystUrlMap.shelve") as urlMap:
            for i in range(count):
                if i >= len(finalPostings):
                    print("----------No more results found----------")
                    break
                urlInfo = urlMap[str(finalPostings[i].docID)]
                print(f"#{i+1}: {urlInfo[0]} ({urlInfo[1]})")
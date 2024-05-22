import os
import json
import psutil
import re
import nltk
import numpy as np
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from transformers import pipeline
from Posting import Posting
import pickle

simHashQueueLength = 100
simHashThreshold = 0.85
batchSize = 15000
readingMemoryLimit = 1000
indexingMemoryLimit = 500
ps = PorterStemmer()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
FOLDERNAME = "ANALYST"

def intOrFloat(s):
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)$'
    return bool(re.fullmatch(pattern, s))

def summarize_text(content):
    try:
        summary = summarizer(content, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return content[:150] + '...'

def read_and_index_json_files(folder_path):
    inverted_index_id = 1
    documents_skipped = 0
    unique_words = set()
    json_set = set()
    corpus = []
    doc_map = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                json_set.add(file_path)
                memory = psutil.virtual_memory()

                if memory.available / (1024 ** 2) < readingMemoryLimit:
                    print(f"Memory limit reached; current memory: {memory.available / (1024 ** 2)} MB; indexing {len(json_set)} documents")
                    inverted_index_id, documents_skipped, unique_words, corpus, doc_map = build_index(json_set, inverted_index_id, documents_skipped, unique_words, corpus, doc_map)
                    json_set.clear()

                if len(json_set) > batchSize:
                    print(f"Batch size reached, indexing {batchSize} documents")
                    inverted_index_id, documents_skipped, unique_words, corpus, doc_map = build_index(json_set, inverted_index_id, documents_skipped, unique_words, corpus, doc_map)
                    json_set.clear()

    if len(json_set) > 0:
        print(f"Indexing remaining {len(json_set)} documents")
        inverted_index_id, documents_skipped, unique_words, corpus, doc_map = build_index(json_set, inverted_index_id, documents_skipped, unique_words, corpus, doc_map)
        json_set.clear()

    print(f"Total documents indexed: {inverted_index_id}. Total documents skipped: {documents_skipped}. Total unique tokens: {len(unique_words)}")

    # Compute TF-IDF manually
    doc_freq = defaultdict(int)
    for doc_tokens in corpus:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    num_docs = len(corpus)
    inverted_index = defaultdict(list)
    for doc_id, doc_tokens in enumerate(corpus):
        term_freq = Counter(doc_tokens)
        for term, count in term_freq.items():
            tf = count / len(doc_tokens)
            idf = np.log(num_docs / (1 + doc_freq[term]))
            tf_idf = tf * idf
            inverted_index[term].append(Posting(doc_id, tf_idf))

    # Write the inverted index to file
    with open(os.join(os.getcwd() + f"{FOLDERNAME}_inverted_index.pkl"), "wb") as f:
        pickle.dump(inverted_index, f)

    return inverted_index_id, documents_skipped, unique_words

def parse_document_into_tokens(json_file):
    content = json_file["content"]
    soup = BeautifulSoup(content, "html.parser")

    if soup:
        try:
            title = soup.find('title').text if soup.find('title') else None
            total_text = soup.get_text()

            if not check_duplicate(soup):
                return "", Counter(), ""

            if len(total_text) != 0:
                summary = summarize_text(total_text)
                tokens = [string for string in nltk.word_tokenize(total_text) if len(string) > 1]
                tokens = [ps.stem(token) for token in tokens if not intOrFloat(token)]
                return title, tokens, summary
            
        except Exception as e:
            print(e)
            return "", Counter(), ""
        
    return "", Counter(), ""
        

def build_index(json_set, inverted_index_id, documents_skipped, unique_words, corpus, doc_map):

    indexed_hashtable = dict()
    map_of_urls = dict()

    for file_path in json_set:

        with open(file_path, "r") as f:

            json_data = json.load(f)
            title, tokens, summary = parse_document_into_tokens(json_data)

            if len(tokens) != 0:
                inverted_index_id += 1
                print(f"Parsing doc {inverted_index_id}")
                for token in tokens:
                    unique_words.add(token)
                    if token not in indexed_hashtable:
                        indexed_hashtable[token] = [inverted_index_id]
                    else:
                        indexed_hashtable[token].append(inverted_index_id)
                
                map_of_urls[str(inverted_index_id)] = [title if title else "Title Not Found", json_data["url"], summary]
                corpus.append(tokens)
                doc_map.append(file_path)

                memory = psutil.virtual_memory()
                if memory.available / (1024 ** 2) < indexingMemoryLimit:
                    print(f"Memory limit reached; current memory: {memory.available / (1024 ** 2)} MB; indexing {inverted_index_id} documents")
                    with open(f"partial_inverted_index_{inverted_index_id}.pkl", "wb") as partial_index_file:
                        pickle.dump(indexed_hashtable, partial_index_file)
                    indexed_hashtable.clear()
            else:
                documents_skipped += 1
                print(f"Skipping {inverted_index_id}")

    print(f"Indexed {len(json_set)} documents. Writing to disk.")

    # Write the partial inverted index to file
    with open(f"{FOLDERNAME}_partial_inverted_index_{inverted_index_id}.pkl", "wb") as partial_index_file:
        pickle.dump(indexed_hashtable, partial_index_file)

    with open(f"{FOLDERNAME}_url_map.pkl", "wb") as url_map_file:
        pickle.dump(map_of_urls, url_map_file)

    return inverted_index_id, documents_skipped, unique_words, corpus, doc_map

def check_duplicate(soup):
    total_text = soup.get_text()
    crc_hash = cyclic_redundancy_check(total_text)

    # Handling hashOfPages.pkl
    hash_file_path = f"{FOLDERNAME}_hashOfPages.pkl"
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "rb") as hash_of_pages_file:
            hash_of_pages = pickle.load(hash_of_pages_file)
    else:
        hash_of_pages = {}

    if str(crc_hash) in hash_of_pages:
        return False
    else:
        hash_of_pages[str(crc_hash)] = True
        with open(hash_file_path, "wb") as hash_of_pages_file:
            pickle.dump(hash_of_pages, hash_of_pages_file)

    # Handling simHashSetLock.pkl
    sim_hash_set_lock_file_path = f"{FOLDERNAME}_simHashSetLock.pkl"
    if os.path.exists(sim_hash_set_lock_file_path):
        with open(sim_hash_set_lock_file_path, "rb") as sim_hash_set_lock_file:
            sim_hash_set_lock = pickle.load(sim_hash_set_lock_file)
    else:
        sim_hash_set_lock = {"Queue": []}

    if "Queue" not in sim_hash_set_lock:
        sim_hash_set_lock["Queue"] = []

    sim_hash = simHash(total_text)
    while len(sim_hash_set_lock["Queue"]) > simHashQueueLength:
        sim_hash_set_lock["Queue"].pop(0)
        
    for sim in sim_hash_set_lock["Queue"]:
        if areSimilarSimHashes(sim_hash, sim, simHashThreshold):
            return False

    sim_hash_set_lock["Queue"].append(sim_hash)

    with open(sim_hash_set_lock_file_path, "wb") as sim_hash_set_lock_file:
        pickle.dump(sim_hash_set_lock, sim_hash_set_lock_file)

    return True

def cyclic_redundancy_check(page_data):
    crc_hash = 0xFFFF

    for byte in page_data.encode():
        crc_hash ^= byte

        for _ in range(8):
            if crc_hash & 0x0001:
                crc_hash = (crc_hash >> 1) ^ 0xA001
            else:
                crc_hash >>= 1

    return crc_hash ^ 0xFFFF

def simHash(page_data):
    weighted_words = Counter([ps.stem(string) for string in nltk.word_tokenize(page_data) if len(string) > 1])
    hash_values = {word: bit_hash(word) for word in weighted_words}

    simhash_value = [0] * 8

    for word, count in weighted_words.items():
        word_hash = hash_values[word]

        for i in range(8):
            bit = (word_hash >> i) & 1
            simhash_value[i] += (1 if bit else -1) * count
    
    simhash_fingerprint = 0
    for i in range(8):
        if simhash_value[i] > 0:
            simhash_fingerprint |= 1 << i
    
    return simhash_fingerprint

def bit_hash(word):
    hash = 0
    for character in word:
        hash += ord(character)
    return hash % 256

def areSimilarSimHashes(first_simhash, second_simhash, threshold):
    similar_bits = bin(first_simhash ^ second_simhash).count('0')
    similarity = similar_bits / 8

    return similarity >= threshold


def merge_partial_indexes(input_dir, output_file):
    merged_index = {}

    for root, _, files in os.walk(input_dir):
        for file in files:
            with open(os.path.join(root, file), "rb") as index_file:
                partial_index = pickle.load(index_file)
                merge_index(merged_index, partial_index)
            # Remove the files once done
            os.remove(os.path.join(root, file))
            print(f"Removing partial index : {os.path.join(root, file)}")
                
    with open(output_file, "wb") as merged_index_file:
        pickle.dump(merged_index, merged_index_file)

def merge_index(merged_index, partial_index):
    for term, postings in partial_index.items():
        if term in merged_index:
            merged_index[term].extend(postings)
        else:
            merged_index[term] = postings

if __name__ == "__main__":
    current_path = os.getcwd()
    
    inverted_index_id, documents_skipped, unique_words = read_and_index_json_files(current_path + "/" + FOLDERNAME)
    print(f"{FOLDERNAME} data set loaded and indexed")

    # Merging partial indexes into a single file
    inverted_index_path = f"{FOLDERNAME}_inverted_index"
    merged_index_path = f"{FOLDERNAME}_merged_inverted_index.pkl"

    merge_partial_indexes(inverted_index_path, merged_index_path)

    # Cleanup
    if os.path.exists(f"{FOLDERNAME}_info.pkl"):
        os.remove(f"{FOLDERNAME}_info.pkl")
        print(f"File {FOLDERNAME}_info.pkl has been deleted.")

    # Writing metadata to a separate file
    with open(f"{FOLDERNAME}_info.pkl", "wb") as dev_info_file:
        pickle.dump({
            "indexed_documents": inverted_index_id,
            "skipped_documents": documents_skipped,
            "unique_tokens": len(unique_words)
        }, dev_info_file)

    # Cleanup
    if os.path.exists(f"{FOLDERNAME}_inverted_index"):
        os.remove(f"{FOLDERNAME}_inverted_index")
        print(f"{FOLDERNAME}_inverted_index has been deleted.")

    # Cleanup
    if os.path.exists(f"{FOLDERNAME}_hashOfPages.pkl"):
        os.remove(f"{FOLDERNAME}_hashOfPages.pkl")
        print(f"{FOLDERNAME}_hashOfPages.pkl has been deleted.")

    # Cleanup
    if os.path.exists(f"{FOLDERNAME}_simHashSetLock.pkl"):
        os.remove(f"{FOLDERNAME}_simHashSetLock.pkl")
        print(f"{FOLDERNAME}_simHashSetLock.pkl has been deleted.")

    # even more cleanup
    for root, dirs, files in os.walk(current_path):
        for file in files:
            if file.startswith(f"{FOLDERNAME}_partial_inverted_index"):
                os.remove(f"{FOLDERNAME}_simHashSetLock.pkl")
                print(f"{file} has been deleted.")
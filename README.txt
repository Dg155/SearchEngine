# About the Project
This project is a search engine which can read and process data from websites and is able to scale up and compete with major modern search engines
such as Google and Bing. The project is split into two parts, the indexer and the query processor. The indexer is responsible for reading through
all the data from the ANALYST and DEV directories and creating an inverted index. The query processor is responsible for taking in a query from the user
and returning the most relevant documents based on the query. The project is written in Python and uses the Beautiful Soup and NLTK library for text processing,
as well as the kivy library for the GUI.

# How to use
- Running the files may prompt you to install the required libraries. You can do this by running `pip install -r requirements.txt`
### Indexer
indexer.py:
- Running this file through the terminal requires 2 arguments. The first argument allows the user to choose between the ANALYST and DEV directories.
The second argument allows the user to choose the batch size of documents the indexer will process in memory before writing to disk.
- The first argument must be a string and must be either "ANA" or "DEV"
- The second argument must be an integer and must be greater than 0
- Example: `python indexer.py ANA 500`
- Example: `python indexer.py DEV 20000`
### Query Processor
search.py:
- Running this file through the terminal will prompt the user to enter a query, and a number of results to return. No arguments are required.
- Example: `python search.py`
search_app.py:
- Running this file through the terminal will open a GUI where the user can enter a query and a number of results to return.
- Example: `python search_app.py`
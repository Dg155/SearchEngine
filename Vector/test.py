import numpy
import pandas as pd
from towhee import pipe, ops, DataCollection
import csv
import os
import json
from bs4 import BeautifulSoup
from towhee import AutoPipes, AutoConfig


sentence_embedding = AutoPipes.pipeline('sentence_embedding')
title_vector =  sentence_embedding("title").to_list()

# Define the column names and data types according to your schema
column_names = ['id', 'title', 'link', 'content_vector']
column_types = {'id': 'int64', 'title': 'str', 'link': 'str', 'content_vector': 'object'}

df = pd.read_csv(r"C:\Users\kidro\OneDrive\Desktop\School\SearchEngine\Vector\fileInfo.csv", skiprows=[0], names=column_names, dtype=column_types)

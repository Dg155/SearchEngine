import json
import os
from bs4 import BeautifulSoup
from transformers import pipeline
import csv

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
FOLDERNAME = "DEV"

def clean_html(content, encoding):
    """Cleans HTML content by removing scripts, styles, and comments."""
    soup = BeautifulSoup(content, "html.parser", from_encoding=encoding)
    for script_or_style in soup(["script", "style", "noscript", "iframe"]):
        script_or_style.extract()
    return soup

def summarize_text(content):
    try:
        summary = summarizer(content, max_length= 150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return "No summary could be created"

def read_and_index_json_files(folder_path):
    with open(f"{FOLDERNAME}_summaries.csv", mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'title', 'summary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    try:
                        print(f"Summarizing {file}")
                        file_path = os.path.join(root, file)
                        with open(file_path) as jsonFile:
                            jsonContent = json.load(jsonFile)
                            content = jsonContent.get("content", "")
                            encoding = jsonContent.get("encoding", None)
                            
                            cleaned_html = clean_html(content, encoding)
                            title = cleaned_html.find('title').text if cleaned_html.find('title') else "No title found"
                            body = cleaned_html.find('body')
                            total_text = body.get_text() if body else cleaned_html.get_text()
                            if len(total_text.strip()) == 0:
                                print(f"No content found in {file}")
                                continue
                            
                            summary = summarize_text(total_text.strip())
                            url = str(jsonContent.get("url", ""))
                            writer.writerow({'url': url.strip(), 'title': title.strip(), 'summary': summary})
                    except json.JSONDecodeError:
                        print(f"Error reading JSON file: {file}")
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    current_path = os.getcwd()
    read_and_index_json_files(current_path + "/" + FOLDERNAME)

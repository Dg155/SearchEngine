from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(content):
    try:
        if len(content) < 30:
            return content
        
        summary = summarizer(content, max_length=150, min_length=30, do_sample=False)
        
        if len(summary) == 0 or 'summary_text' not in summary[0]:
            return "Failed to generate a summary."
        
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return "Error occurred while summarizing the content."

# Example usage:
content_to_summarize = "Your input text goes here."
print(summarize_text(content_to_summarize))
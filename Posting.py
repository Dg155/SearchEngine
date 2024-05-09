class Posting:
    def __init__(self, docID, freqCount, url, fields = []):
        self.docID = docID
        self.freqCount = freqCount # We would need to change this to TF-IDF eventually
        self.url = url
        self.fields = fields
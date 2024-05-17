class Posting:
    def __init__(self, docID, freqCount, fields = []):
        self.docID = docID
        self.freqCount = freqCount # We would need to change this to TF-IDF eventually
        self.fields = fields
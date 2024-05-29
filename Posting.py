class Posting:
    def __init__(self, docID, freq, fields = [], tf = 0, idf = 0):
        self.docID = docID
        self.freq = freq
        self.fields = fields
        self.tf = tf
        self.idf = idf
        self.tfidf = tf * idf
class Posting:
    def __init__(self, docID, freq, boldCount, headerCount, titleCount, tf = 0, idf = 0):
        self.docID = docID
        self.freq = freq
        self.boldCount = boldCount
        self.headerCount = headerCount
        self.titleCount = titleCount
        self.tf = tf
        self.idf = idf
        self.tfidf = tf * idf
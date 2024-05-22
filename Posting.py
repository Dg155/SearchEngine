class Posting:
    def __init__(self, docID, tf_idf, fields = []):
        self.docID = docID
        self.tf_idf = tf_idf # We would need to change this to TF-IDF eventually
        self.fields = fields
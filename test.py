from nltk.stem import PorterStemmer
import re
ps = PorterStemmer()

def is_int_or_float(s):
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)$'
    return bool(re.fullmatch(pattern, s))

print([is_int_or_float(token) for token in ["1.619", "-0.4636", "sharker", "0.04671", "-0.09881"]])
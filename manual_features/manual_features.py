import re

ALL_CAPS = re.compile(r'^[A-Z]{2,}$') # all cap words
ELONGATED = re.compile(r'([a-z])\1{2,}') # elongated words
QE_MARKS = re.compile(r'(\?|!){2,}') # question or exclamation marks
ENDING_WITH_QE = re.compile(r'((.*\?)|(.*!))$') # ending with question or exclamation mark

def count_pattern(text, pattern):
    count = 0
    for word in text:
        if pattern.search(word):
            count += 1
    return count

def count_punctuations(text, pattern):
    counts = []
    for word in text:
        res = [ m.end() - m.start() for m in pattern.finditer(word)]
        counts.append(sum(res))
    return sum(counts)

def ending_punctuation(text, pattern):
    return bool(pattern.match(text[-1]))*1

def repetition_score(text):
    scores = []
    for word in text:
        scores.append(1 - len(set(word))/len(word))
    return max(scores)
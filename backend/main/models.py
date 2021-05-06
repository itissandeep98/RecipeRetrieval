from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import json
import string
import numpy as np
from collections import defaultdict, Counter
from tqdm.notebook import tnrange
import pickle


nltk.download("punkt")
nltk.download("stopwords")


class tf_idfmatrices():
    def __init__(self, DFpostings, tokens, docs_count):
        self.DFpostings = DFpostings
        self.tokens = tokens
        self.docs_count = docs_count
        self.vocab_count = len(DFpostings)
        self.vocabulary = [x for x in DFpostings]
        self.idf = dict()
        self.counter_lists = []
        self.tf_idf_TermFreq = np.zeros((docs_count, self.vocab_count))

    def generateIDF(self):
        for key in self.DFpostings:
            doc_freq = len(self.DFpostings[key])
            self.idf[key] = np.log10((self.docs_count/doc_freq) + 1)

    def generateCounterLists(self):
        for i in tnrange(self.docs_count):
            self.counter_lists.append(Counter(self.tokens[i]))

    def generateTermFreq(self):
        for i in tnrange(len(self.vocabulary)):
            for j in range(self.docs_count):
                self.tf_idf_TermFreq[j][i] = (
                    self.counter_lists[j][self.vocabulary[i]]/len(self.tokens[j]))*self.idf[self.vocabulary[i]]


def stripSpecialChar(text):
    return ''.join(ch for ch in text if ch.isalnum() and not ch.isdigit() and ch not in string.punctuation)


def preProcess(text):
    stemmer = SnowballStemmer("english")
    stopWords = set(stopwords.words('english'))

    # convert all text to lower case
    text = text.lower()
    # tokenizing the text
    text_tokens = word_tokenize(text)

    stemmedWords = list([stemmer.stem(word) for word in text_tokens])
    validTokens = [i for i in stemmedWords if i not in stopWords]

    # stripping special characters
    validTokens = [stripSpecialChar(x) for x in validTokens]
    # Choosing only words which has length > 1
    validTokens = [x for x in validTokens if len(x) > 1]
    return set(validTokens)


def getResults(sentence_query):
    listofwords = preProcess(sentence_query)
    query_eval = np.zeros((title_tf_idf_obj.docs_count, 1))
    getscore(title_tf_idf_obj, listofwords)
    getscore(file_tf_idf_obj, listofwords)

    ans = Top5(query_eval[:, 0])
    f = open('main/Data/Final_mapping.json',)
    d = json.load(f)

    for i in ans:
        file1 = open(d[i], "r")
        print(d[i][15:-4])
        print(file1.readline())


def getscore(objtype, listofwords):
    indextolookfor = []
    for word in listofwords:
        # instead of forming a query vector, we just extracted the indices of the querytokens
        index = objtype.vocabulary.index(word)
        indextolookfor.append(index)
    for docs in range(objtype.docs_count):
        for query in indextolookfor:
            query_eval[docs][0] += (objtype.tf_idf_TermFreq[docs][query])


def Top5(alist):
    return sorted(range(len(alist)), key=lambda i: alist[i], reverse=True)[:5]



class InvertedIndex():
    def __init__(self):
        self.DFpostings = {}  # Dictionary contains Posting List for each words
        self.termsInFile = []  # List contains all words in a single file

    def stripSpecialChar(self, text):
        return ''.join(ch for ch in text if ch.isalnum() and not ch.isdigit() and ch not in string.punctuation)

    def preProcess(self, text):
        stemmer = SnowballStemmer("english")
        stopWords = set(stopwords.words('english'))

        # convert all text to lower case
        text = text.lower()
        # tokenizing the text
        text_tokens = word_tokenize(text)

        stemmedWords = list([stemmer.stem(word)
                            for word in text_tokens])   # Stemming
        # Removing Stop Words
        validTokens = [i for i in stemmedWords if i not in stopWords]

        # stripping special characters
        validTokens = [self.stripSpecialChar(x) for x in validTokens]
        # Choosing only words which has length > 1
        validTokens = [x for x in validTokens if len(x) > 1]
        return validTokens, set(validTokens)

    def indexFile(self, file, fileId):
        '''
        Creates Index for each File
        '''
        tokens, setTokens = self.preProcess(file)
        self.termsInFile.append(tokens)
        for i in setTokens:
            if i in self.DFpostings:
                self.DFpostings[i].append(fileId)
            else:
                self.DFpostings[i] = [fileId]

    def save(self, file):
        '''
        Save the index to a file locally
        '''
        json.dump(self.DFpostings, open("DFPostings"+file, "w"))
        json.dump(self.termsInFile, open("TermsInfile"+file, "w"))



# with open('main/Data/ProjectFile.obj', 'rb') as file_object:
#     raw_data = file_object.read()
# file_obj = pickle.loads(raw_data)

# with open('main/Data/ProjectTitle.obj', 'rb') as file_object:
#     raw_data = file_object.read()
# title_obj = pickle.loads(raw_data)

# with open('main/Data/TF_IDF_Calculated_File.obj', 'rb') as file_object:
#     raw_data = file_object.read()
# file_tf_idf_obj = pickle.loads(raw_data)

# with open('main/Data/TF_IDF_Calculated_Title.obj', 'rb') as file_object:
#     raw_data = file_object.read()
# title_tf_idf_obj = pickle.loads(raw_data)

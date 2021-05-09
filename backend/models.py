from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import json
import string
import numpy as np
from collections import Counter
from tqdm.notebook import tnrange


# nltk.download("punkt")
# nltk.download("stopwords")


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


class Results():
    def __init__(self, file_obj, title_obj,
                 file_tf_idf_obj, title_tf_idf_obj):
        self.file_obj = file_obj
        self.title_obj = title_obj
        self.file_tf_idf_obj = file_tf_idf_obj
        self.title_tf_idf_obj = title_tf_idf_obj

    def stripSpecialChar(self, text):
        return ''.join(ch for ch in text if ch.isalnum() and not ch.isdigit() and ch not in string.punctuation)

    def preProcess(self, text):
        stemmer = SnowballStemmer("english")
        stopWords = set(stopwords.words('english'))

        # convert all text to lower case
        text = text.lower()
        # tokenizing the text
        text_tokens = word_tokenize(text)

        stemmedWords = list([stemmer.stem(word) for word in text_tokens])
        validTokens = [i for i in stemmedWords if i not in stopWords]

        # stripping special characters
        validTokens = [self.stripSpecialChar(x) for x in validTokens]
        # Choosing only words which has length > 1
        validTokens = [x for x in validTokens if len(x) > 1]
        return set(validTokens)

    def getscore(self, objtype, listofwords, query_eval):
        indextolookfor = []
        for word in listofwords:
            # instead of forming a query vector, we just extracted the indices of the querytokens
            index = objtype.vocabulary.index(word)
            indextolookfor.append(index)
        for docs in range(objtype.docs_count):
            for query in indextolookfor:
                query_eval[docs][0] += (objtype.tf_idf_TermFreq[docs][query])

    def Top5(self, alist):
        return sorted(range(len(alist)), key=lambda i: alist[i], reverse=True)[:5]

    def getResults(self, sentence_query):
        listofwords = self.preProcess(sentence_query)
        query_eval = np.zeros((self.title_tf_idf_obj.docs_count, 1))
        self.getscore(self.title_tf_idf_obj, listofwords, query_eval)
        self.getscore(self.file_tf_idf_obj, listofwords, query_eval)

        ans = self.Top5(query_eval[:, 0])
        mapping = json.load(open('Data/Final_mapping.json',))
        data = []

        for i in ans:
            file1 = open(mapping[i], "r")
            fileid = mapping[i][15:-4]
            recipe = file1.readline()
            data.append({"key": fileid, "title": recipe,"message":"test"})
        return data


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

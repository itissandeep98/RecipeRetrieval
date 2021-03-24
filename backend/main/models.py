from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import json
import string
import numpy as np
from collections import defaultdict

nltk.download("punkt")
nltk.download("stopwords")


class Response():
    def __init__(self, data):
        self.data = set(data)

    def getMapping(self, file):
        '''
         Prints out the corresponding document names from list of document IDs
        '''
        self.mapping = json.load(open(file))
        # list of tuples consisting of document id and their location
        data = list(map(lambda i: (i, self.mapping[i]), self.data))
        data.sort()
        return data

    def __str__(self):
        return str(len(self.data))

    def add(self, response):
        '''
        Performs the set union of given object and pass object
        '''
        return Response(set.union(self.data, response.data))

    def intersect(self, response):
        '''
        Performs the set intersection of given object with pass object
        '''
        return Response(set.intersection(self.data, response.data))

    def diff(self, response):
        '''
        Performs the set difference of given object with pass object
        '''
        return Response(set.difference(self.data, response.data))


class Query():
    def __init__(self, file):
        '''
        initializes the object with loading the index file
        '''
        self.db = json.load(open(file))
        self.db = defaultdict(lambda: [], self.db)

    def OR(self, term1, term2):
        '''
        Finds the docs after applying OR operation on given list of documents
        '''
        return term1.add(term2)

    def AND(self, term1, term2):
        '''
        Finds the docs after applying AND operation on given list of documents
        '''
        return term1.intersect(term2)

    def ANDNOT(self, term1, term2):
        '''
        Finds the docs after applying AND NOT operation on given list of documents
        '''
        univ = Response(np.arange(467))
        not_term2 = univ.diff(term2)
        return term1.intersect(not_term2)

    def ORNOT(self, term1, term2):
        '''
        Finds the docs after applying OR NOT operation on given list of documents
        '''
        univ = Response(np.arange(467))
        not_term2 = univ.diff(term2)
        joint = term1.intersect(term2)
        return not_term2.add(joint)

    def count(self, first, second):
        i, j, count = 0, 0, 0
        while(i < len(first) and j < len(second)):
            count += 1
            if(first[i] < second[j]):
                i += 1
            elif(first[i] > second[j]):
                j += 1
            else:
                i += 1
                j += 1
        return count

    def no_comparisonsOR(self, term1, term2):
        '''
        To return the number of comparisons it will make in OR operations between two list of documents
        '''
        first = list(term1.data)
        first.sort()
        second = list(term2.data)
        second.sort()
        return self.count(first, second)

    def no_comparisonsAND(self, term1, term2):
        '''
        To return the number of comparisons it will make in AND operations between two list of documents
        '''
        first = list(term1.data)
        first.sort()
        second = list(term2.data)
        second.sort()
        return self.count(first, second)

    def no_comparisonsANDNOT(self, term1, term2):
        '''
        To return the number of comparisons it will make in AND NOT operations between two list of documents
        '''
        first = list(term1.data)
        first.sort()
        univ = Response(np.arange(467))
        not_term2 = univ.diff(term2)
        second = list(not_term2.data)
        second.sort()
        return self.count(first, second)

    def no_comparisonsORNOT(self, term1, term2):
        '''
        To return the number of comparisons it will make in OR NOT operations between two list of documents
        '''
        first = list(term1.data)
        first.sort()
        univ = Response(np.arange(467))
        not_term2 = univ.diff(term2)
        second = list(not_term2.data)
        second.sort()

        return self.count(first, second)

    def stripSpecialChar(self, text):
        return ''.join(ch for ch in text if ch.isalnum() and not ch.isdigit() and ch not in string.punctuation)

    def preProcess(self, text):
        stemmer = SnowballStemmer("english")
        stopWords = set(stopwords.words('english'))

        # convert all text to lower case
        text = text.lower()
        # tokenizing the text
        text_tokens = word_tokenize(text)

        # stemmedWords = list([stemmer.stem(word) for word in text_tokens])
        # validTokens = [i for i in stemmedWords if i not in stopWords]

        # removing stop words
        validTokens = [i for i in text_tokens if i not in stopWords]

        # stripping special characters
        validTokens = [self.stripSpecialChar(x) for x in validTokens]
        # Choosing only words which has length > 1
        validTokens = [x for x in validTokens if len(x) > 1]
        return validTokens

    def processQuery(self, inp, ops):
        '''
        Performs query with given string and list of operations
        '''
        terms = self.preProcess(inp)
        output = Response(self.db[terms[0]])
        comparisons = 0
        for i in range(1, len(terms)):
            curr = Response(self.db[terms[i]])
            if(ops[i-1] == 'OR'):
                output = self.OR(output, curr)
                comparisons += self.no_comparisonsOR(output, curr)
            elif(ops[i-1] == 'AND'):
                output = self.AND(output, curr)
                comparisons += self.no_comparisonsAND(output, curr)
            elif(ops[i-1] == 'OR NOT'):
                output = self.ORNOT(output, curr)
                comparisons += self.no_comparisonsORNOT(output, curr)
            elif(ops[i-1] == 'AND NOT'):
                output = self.ANDNOT(output, curr)
                comparisons += self.no_comparisonsANDNOT(output, curr)
            else:
                raise Exception("Operand not Identified:"+ops[i-1])

        return output, comparisons

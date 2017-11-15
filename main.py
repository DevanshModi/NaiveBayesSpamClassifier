#TODO: will paths work properly?

############################################################
# Imports
############################################################

import email
import math
import os
import copy
import operator


############################################################
# Section 1: Spam Filter
############################################################

#TODO: edge cases
#returns [] of tokens
def load_tokens(email_path):

    f = open(email_path, 'r')
    obj = email.message_from_file(f)
    #print obj
    x = []
    for line in email.Iterators.body_line_iterator(obj):
        x+=line.split()
        #print x

    return x

#TODO: precision error, add float cast, optimize it to make it faster

def log_probs(email_paths, smoothing):
    probs = dict()
    vocab = dict()

    for path in email_paths:
        tokens = load_tokens(path)

        for word in tokens:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    sumvalues = sum(vocab.values())
    count = len(vocab)
    svalue = 1e-5

    for word in vocab.keys():
        if len(word.split()) != 2:
            numerator = float(vocab[word] + smoothing)

            denom = float(sumvalues) + (count+float(1))*smoothing

            val = float(numerator) / float(denom)

            probs[word] = float(math.log(val))
        else:
            numerator = float(vocab[word] + svalue)

            denom = float(sumvalues) + (count+float(1))*svalue

            val = float(numerator) / float(denom)

            probs[word] = float(math.log(val))

    numerator = float(smoothing)
    denom = float(sumvalues + (smoothing * (count + 1)))
    probs["<UNK>"] = math.log(numerator/denom)

    return probs

class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        #Spam
        self.spamPaths = os.listdir(spam_dir)

        self.spamPaths = [spam_dir + '/' + name for name in self.spamPaths]

        #Vocab for Spam here
        self.spamDict = log_probs(self.spamPaths, smoothing)

        # Ham
        self.hamPaths = os.listdir(ham_dir)

        self.hamPaths = [ham_dir + '/' + name for name in self.hamPaths]

        self.hamDict = log_probs(self.hamPaths, smoothing)

        #Class values - spam/!spam

        self.Spam = float(len(self.spamPaths)) / float(len(self.spamPaths) + len(self.hamPaths))

        self.NotSpam = float(len(self.hamPaths)) / float(len(self.spamPaths) + len(self.hamPaths))


    #TODO: Add log calculation
    def is_spam(self, email_path):
        emailwords = dict()
        tokens = load_tokens(email_path)

        #Get the count for all tokens
        for word in tokens:
            if word not in emailwords:
                emailwords[word] = [1, 0]

            else:
                emailwords[word][0] += 1

        #Set the relevant probabilities
        for word in emailwords.keys():
            if word in self.spamDict:
                emailwords[word][1] = self.spamDict[word]
            else:
                emailwords[word][1] = self.spamDict["<UNK>"]


        spamtotal = [emailwords[i][1]*emailwords[i][0] for i in emailwords.keys()]
        #print spamtotal
        spamtotal = sum(spamtotal)

        sprob = self.Spam * spamtotal
        #print sprob

        emailwords_h = copy.deepcopy(emailwords)

        for word in emailwords_h.keys():
            if word in self.hamDict:
                emailwords_h[word][1] = self.hamDict[word]
            else:
                emailwords_h[word][1] = self.hamDict["<UNK>"]

        hamtotal = [emailwords_h[i][1] * emailwords_h[i][0] for i in emailwords_h.keys()]
        hamtotal = sum(hamtotal)

        hprob = self.NotSpam * hamtotal

        #print "Sprob is" + str(sprob) + "Hprob is" + str(hprob)
        return sprob > hprob

    def most_indicative_spam(self, n):
        temp = dict()

        for word in self.spamDict:
            if word in self.hamDict and word not in temp.keys():
                val1 = math.exp(self.spamDict[word])
                val2 = math.exp(self.hamDict[word])
                pw = self.Spam*val1 + self.NotSpam*val2
                val = self.spamDict[word] - math.log(pw)
                temp[word] = val

        revtemp = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)

        #print revtemp
        #return revtemp[:n]
        res = [x[0] for x in revtemp]

        return res[:n]

    def most_indicative_ham(self, n):
        temp = dict()

        for word in self.hamDict:
            if word in self.spamDict and word not in temp.keys():
                val1 = math.exp(self.spamDict[word])
                val2 = math.exp(self.hamDict[word])
                pw = self.Spam*val1 + self.NotSpam*val2
                val = self.hamDict[word] - math.log(pw)
                temp[word] = val

        revtemp = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)

        #print revtemp
        #return revtemp[:n]
        res = [x[0] for x in revtemp]

        return res[:n]

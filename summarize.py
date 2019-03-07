import nltk
import operator
import math

from nltk.corpus import stopwords  #set NLTK_DATA environment variable to ./nltk/nltk_data
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer

import text_file

def summarize(value):
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    stopWords = set(nltk.corpus.stopwords.words("english"))
    text = text_file.get_text
    words = nltk.tokenize.word_tokenize(text)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue

        word = stemmer.stem(word)

        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = nltk.tokenize.sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq
        sentenceValue[sentence] = sentenceValue[sentence]/math.pow((len(nltk.tokenize.word_tokenize(sentence))),0.4)

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    iteration = 0
    factor = 1.0

    while True:
        summary = ''
        max_chars = 8000
        max_words = 200 #alexa speaks at about 160 wpm
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (factor * average)) and len(sentence.split(" ")) > 3:
                summary += " " + sentence
        if len(summary) < max_chars and len(summary.split(" ")) < max_words:
            break
        iteration += 1
        factor += 0.01

    return summary

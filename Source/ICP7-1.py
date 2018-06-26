from bs4 import BeautifulSoup
import urllib.request
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk import ne_chunk
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

html = ("https://en.wikipedia.org/wiki/Python_(programming_language)")
source_code = urllib.request.urlopen(html)
soup = BeautifulSoup(source_code,'html.parser')
input_text = soup.get_text()

#file = open("input.txt", "w")
#for line in input_text:
    #print(line)
    #file.write(line)


#Tokenization.........................................................
# Breaking a stream of text into words or phrases
words = word_tokenize(input_text) # word tokenizing
sentences = sent_tokenize(input_text)# sentence tokenizing
filterd_words = [] # to find unique words and sentences
filterd_sentences = []

for w in words:
    if w not in filterd_words:
        filterd_words.append(w)
for s in sentences:
    if s not in filterd_sentences:
        filterd_sentences.append(s)
print ("Tokenization...............")
print (filterd_sentences)
print (filterd_words)

#Stemming ....................................................................
# Process of reducing injected words into their stem, base root form
ps = PorterStemmer() # creating object for stemming
stemen = [] # list for storing stemmed words
for w in filterd_words:
    stemen.append(ps.stem(w))
print ("stemming............")
print (stemen)

# POS...........................................................................
# Shows the different parts of speech
pos = nltk.pos_tag(words) # part of speech tagging
print ("POS..........")
print (pos)

#Lemmatization...................................................................
# Process for determing the part of speech of a word and applying different normalization rules
lemmatizer = WordNetLemmatizer ()
lemma =[] # List to store lemmatized words
for w in words:
    lemma.append(lemmatizer.lemmatize(w))
print ("Lemmatization.......")
print (lemma)

# Trigram............................................................................
# Splitting the sentence into three
trigrams=ngrams(words,3)
tri = [] # list to store tuples of three elements
for i in trigrams:
    tri.append (i)
print ("Trigram.......")
print (tri)

# named entity recognision..........................................................

namEnt = nltk.ne_chunk(pos)
print ("Named entity recognition........")
print (namEnt)




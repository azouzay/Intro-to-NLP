#### into to NLTK then small project

#  Introduction to NLTK
# 
## In part 1 of this project, we will use nltk to explore the Herman Melville novel Moby Dick.
##Then in part 2 we will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

import nltk
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

# to work with the raw text we can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
#  to work with the novel in nltk.Text format we can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)
 
# How many tokens (words and punctuation symbols) are in text1?
tokens_text1= len(nltk.word_tokenize(moby_raw)) 

# How many unique tokens (unique words and punctuation) does text1 have?
unique_text1= len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

# After lemmatizing the verbs, how many unique tokens does text1 have?
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]
lemm_text1= len(set(lemmatized))
 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
diversity_text1= tokens_text1/unique_text1

# What percentage of tokens is 'whale'or 'Whale'?
dist = FreqDist(moby_tokens)
    
Whale= (dist['whale']+dist['Whale'])/sum(dist.values())*100

# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
from collections import Counter
import re
20most= Counter(text1).most_common(20)
 
# What tokens have a length of greater than 5 and frequency of more than 150?
dist = FreqDist(moby_tokens)
greater5= sorted ([w for w in set(moby_tokens) if (len(w) > 5) and (dist[w]>150)])

# Find the longest word in text1 and that word's length.
words=[(w, len(w)) for w in moby_tokens]
longest= sorted(words,key=lambda x: x[1], reverse=True)[0]
 
# What unique words have a frequency of more than 2000? What is their frequency?
freq2000_1= [(dist[w],w) for w in set(moby_tokens) if w.isalpha() and (dist[w]>2000)]
freq2000=sorted(freq2000_1, reverse=True)

# What is the average number of tokens per sentence?
sentences = nltk.sent_tokenize(moby_raw)
average-sentences= np.average(result= [len(nltk.word_tokenize(w)) for w in sentences])
 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
wordtags= nltk.pos_tag(moby_tokens)
tag_freq = nltk.FreqDist(tag for (word, tag) in wordtags)
speech= [(tag,tag_freq[tag]) for tag in tag_freq]
speech_5= sorted(result,key=lambda x: x[1], reverse=True)[:5]

# ## Part 2 - Spelling Recommender
# 
from nltk.corpus import words

correct_spellings = words.words()
 
# Provide recommendations for the three default words provided above using the following distance metric:
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**

from sklearn.metrics import jaccard_similarity_score
from nltk import ngrams
from nltk import jaccard_distance

entries=['cormulent', 'incendenece', 'validrate']
    
cormulent_score = [(jaccard_distance(set(ngrams(entries[0], n=3)), set(ngrams(w, n=3))), w) for w in correct_spellings if w[0]=='c']
incendenece_score = [(jaccard_distance(set(ngrams(entries[1], n=3)), set(ngrams(w, n=3))), w) for w in correct_spellings if w[0]=='i']
validrate_score = [(jaccard_distance(set(ngrams(entries[2], n=3)), set(ngrams(w, n=3))), w) for w in correct_spellings if w[0]=='v']
    
recommendation= sorted(cormulent_score)[0][1], sorted(incendenece_score)[0][1],sorted(validrate_score)[0][1]]

# provide recommendations for the three default words provided above using the following distance metric:
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
    
cormulent_score = [(jaccard_distance(set(ngrams(entries[0], n=4)), set(ngrams(w, n=4))), w) for w in correct_spellings if w[0]=='c']
incendenece_score = [(jaccard_distance(set(ngrams(entries[1], n=4)), set(ngrams(w, n=4))), w) for w in correct_spellings if w[0]=='i']
validrate_score = [(jaccard_distance(set(ngrams(entries[2], n=4)), set(ngrams(w, n=4))), w) for w in correct_spellings if w[0]=='v']
    
recommendation2= sorted(cormulent_score)[0][1], sorted(incendenece_score)[0][1],sorted(validrate_score)[0][1]

#  provide recommendations for the three default words provided above using the following distance metric:
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**

from nltk import edit_distance

cormulent_score = [(edit_distance(entries[0], w, transpositions=True),w) for w in correct_spellings if w[0]=='c']
incendenece_score = [(edit_distance(entries[1], w, transpositions=True),w) for w in correct_spellings if w[0]=='i']
validrate_score = [(edit_distance(entries[2], w, transpositions=True),w) for w in correct_spellings if w[0]=='v']
    
recommendation3= sorted(cormulent_score)[0][1], sorted(incendenece_score)[0][1],sorted(validrate_score)[0][1]

#---------

# Project 4- Document Similarity & Topic Modelling

# ## Part 1 - Document Similarity
# The following functions are provided:
# * **`convert_tag:`** converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. You will need to use this function in `doc_to_synsets`.
# * **`document_path_similarity:`** computes the symmetrical path similarity between two documents by finding the synsets in each document using `doc_to_synsets`, then computing similarities using `similarity_score`.
# 
# to write the following functions:
# * **`doc_to_synsets:`** returns a list of synsets in document.
# * **`similarity_score:`** returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2).

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
       
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.
    """
    

    token= nltk.word_tokenize(doc)
    
    postag= nltk.pos_tag(token)
    tags = [tag[1] for tag in postag]
    wordnet_tag = [convert_tag(tag) for tag in tags]
    
    TW=list(zip(token,wordnet_tag))
    sets = [wn.synsets(x,y) for x,y in TW]
    
    
    return [val[0] for val in sets if len(val) > 0]


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.
"""
    s =[]
    for c1 in s1:
        r = []
        scores = [x for x in [c1.path_similarity(c2) for c2 in s2] if x is not None]
        if scores:
            s.append(max(scores))
    return sum(s)/len(s)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2



# `paraphrases` is a DataFrame which contains the following columns: `Quality`, `D1`, and `D2`.
# 
# `Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another(1 for paraphrase, 0 for not paraphrase).
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()

# ### most_similar_docs
p=paraphrases
result=[document_path_similarity(p.D1[i],p.D2[i])for i in range(0,len(p))]
a=np.argmax(result)
most_similar_docs=  (p.D1[a],p.D2[a],document_path_similarity(p.D1[a],p.D2[a]))
most_similar_docs()

# ### label_accuracy
# 
# Provide labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.
# 
# *This function should return a float.*

# In[37]:
from sklearn.metrics import accuracy_score
p=paraphrases
p['label']=0
for i in range(0,len(p)):
    if (document_path_similarity(p.D1[i],p.D2[i]))> 0.75:
        p['label'][i]=1  
label_accuracy= accuracy_score(p['label'], p['Quality'])

# ## Part 2 - Topic Modelling
# using Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`.
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())


# Use the gensim.models.ldamodel.LdaModel constructor to estimate   
#gensim.models.ldamodel.LdaModel   """estimate LDA model parameters on corpus"""

ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word=id_map, num_topics=10, passes=25, random_state=34)


# ### lda_topics
# 
# Using `ldamodel`, find a list of the 10 topics and the most significant 10 words in each topic. This should be structured as a list of 10 tupleswhere each tuple takes on the form:
# `(9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')`  
lda_topics= ldamodel.print_topics(num_topics=10, num_words=10)

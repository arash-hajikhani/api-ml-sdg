#this is the cleaning function i want to be applied for new text submitted for ML model.

import pandas as pd
import string
import textblob
import nltk
import gensim
import numpy as np
from textblob import TextBlob, Word
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def cleaning (df1 , Texteng) :
    
    #df_name : data frame 
    # text_col : text col as series 
    
    #convert the col to list 
    text = df1[Texteng].tolist()
    
    ############################### Basic cleaning ###############################################
    def basicCleaning(abstractLst):
        processed = []
        translator = str.maketrans('', '', string.punctuation)
        for item in abstractLst:
            the_data = item
            the_data = " ".join([x for x in the_data.split(" ") if not x.isdigit()])
            the_data = ' '.join(s for s in the_data.split() if not any(c.isdigit() for c in s))  #terms containing a number removed
            the_data = " ".join([x for x in the_data.split(" ") if not x.isdigit()]) #terms consisting of only numbers removed
            the_data = the_data.translate(translator)            #remove punctuations     
            #the_data = " ".join([wnl.lemmatize(i) for i in the_data.split(" ")])
            the_data = the_data.lower()
            #the_data = ' '.join([word for word in the_data.split() if word not in stopWords_list])
            processed.append(the_data)
        return processed
    
    # apply basic cleaning
    clean_text = basicCleaning(text)
#     print('starting the customized - stopword removal.')    
    ############################## lemmitazation #####################################
    def my_lemmitazation (abstarct_list):
        # first run pos tagging, then lemmatized all forms of speech tagged terms
        lemms_list =[]
        for sentence in abstarct_list:
            sent = TextBlob(sentence)
            #tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN, "V": wordnet.VERB,"R": wordnet.ADV}
            tag_dict = {"J": 'a',  "N": 'n', "V": 'v', "R": 'r'}
            words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]  
            
            #only keep verbs and nouns
            keep =  [(word, tag) for word, tag in words_and_tags if tag in ('a', 'n' ,'v')]
            lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
            temp = " ".join(lemmatized_list)
            lemms_list.append(temp)        
        return lemms_list  
    
    #apply lemmitization
    clean_text_custop_lemmitize = my_lemmitazation(clean_text)
#     print('lemmitization - done.')
    

    # add new col and store clean text
    df1.loc[: , 'cleaned_text'] = clean_text_custop_lemmitize
    
#     print('all cleaning tasks are done.')
    return df1




def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.vectors_norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens

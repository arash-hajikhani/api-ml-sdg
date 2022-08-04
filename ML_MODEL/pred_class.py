from base64 import encode
import encodings
import pickle
import numpy as np
import pandas as pd
import warnings
import os
import ML_MODEL.preprocessing as preprocess
warnings.filterwarnings('ignore')
import gensim
from sys import platform

class prediction:
    def __init__(self):
        self.wv = gensim.models.KeyedVectors.load_word2vec_format("ML_MODEL/word2vec-google-news-300.gzip", binary=True)
        self.wv.init_sims(replace=True)
        self.filename = 'ML_MODEL/word2vec-logreg_SDG-ML-model3.sav'
        self.loaded_model = pickle.load(open(self.filename, 'rb'))
        
    
    def get_predictions(self,df,key_column,text_column):
        final_data = preprocess.cleaning(df , text_column)
        New_text_tokenized = final_data.apply(lambda r: preprocess.w2v_tokenize_text(r['cleaned_text']), axis=1).values
        New_text_word_average = preprocess.word_averaging_list(self.wv,New_text_tokenized)
        ynew = self.loaded_model.predict_proba(New_text_word_average)
        probability_table = pd.DataFrame(ynew, columns = self.loaded_model.classes_, index=final_data.EID)
        nlargest = 5

        probability_table = probability_table.round(3) * 100
        order = np.argsort(-probability_table.values, axis=1)[:, :nlargest]
        prob_order = -np.sort(-probability_table.values)[:, :nlargest]
        top3class_result = pd.DataFrame(probability_table.columns[order],
                          columns=['Machine_guess_{}'.format(i) for i in range(1, nlargest+1)],
                                   index = probability_table.index
                                   ) 

        top3prob_result = pd.DataFrame(prob_order, columns=['Machine_guess_{}'.format(i) for i in range(1, nlargest+1)],
                                     index = probability_table.index
                                      )

        top3_classprob_result =  top3class_result.astype(str).add('= ').add(round(top3prob_result,2).astype(str)).add('%')

        top3_classprob_result.reset_index(inplace=True)

        for i in range(5):
            col = f"Machine_guess_{i+1}"
            final_data.insert(i+3,col , top3_classprob_result[col])

        return final_data
import math
import re
import string
from collections import defaultdict

import gensim
import numpy as np
import pandas as pd
import time

from nltk import SnowballStemmer
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
from textblob import Word
from tqdm import tqdm


class DataFramePreprocesser:
    def __init__(self,
                 treatment_col='treatment',
                 outcome_col='outcome',
                 text_col=None,
                 include_cols=None,
                 ignore_cols=None,
                 verbose=1,
                 word_vec=False):
        """
        Instantiates the DataframePreprocessor instance.
        """
        if include_cols is None:
            include_cols = []
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.text_col = text_col
        self.include_cols = include_cols
        self.ignore_cols = ignore_cols
        self.v = verbose
        self.word_vec=word_vec
        self.word2vec_model = Word2Vec.load("word2vec/word2vecgensimmodel/w2v_features_10minwordcounts_10context.model")
        # self.model=Word2Vec(list(df[self.text_col]), window=10, min_count=2)
        self.stop = stopwords.words('english')
        # these variables set by preprocess
        self.feature_names = None
        self.feature_types = {}
        self.cat_dict = {}
        self.tv = None
        self.is_classification = None
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def preprocess(self, df,
                   training=False,
                   min_df=0.05,
                   max_df=0.5,
                   ngram_range=(1, 1),
                   stop_words='english',
                   na_cont_value=-1, na_cat_value='MISSING'):

        df = df.dropna()
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pandas DataFrame')
        df = df.rename(columns=lambda x: x.strip())  # strip headers
        # check and re-order test DataFrame
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip data
        df, _ = self._preprocess_column(df, self.treatment_col, is_treatment=True)
        if training:
            df, self.is_classification = self._preprocess_column(df, self.outcome_col, is_treatment=False)
            self.feature_names = [c for c in df.columns.values  \
                                  if c not in [self.treatment_col,
                                               self.outcome_col, self.text_col] + self.ignore_cols]
            for c in self.feature_names:
                self.feature_types[c] = self._check_type(df, c)['dtype']

        X = df[self.feature_names].copy()
        Y = df[self.outcome_col].copy()
        T = df[self.treatment_col].copy()

        # step 2: fill empty values on x
        for c in self.feature_names:
            dtype = self.feature_types[c]
            if dtype == 'string': X[c] = X[c].fillna(na_cat_value)
            if dtype == 'numeric': X[c] = X[c].fillna(na_cont_value)

        # step 3: one-hot encode categorial features
        for c in self.feature_names:
            if c == self.text_col: continue
            if self.feature_types[c] == 'string':
                if training:
                    self.cat_dict[c] = sorted(X[c].unique())
                    catcol = X[c]
                else:
                    catcol = X[c].astype(pd.CategoricalDtype(categories=self.cat_dict[c]));

                X = X.merge(pd.get_dummies(catcol, prefix=c,
                                           drop_first=False),
                            left_index=True, right_index=True)
                del X[c]
        # step 4: for text-based confounder, use extracted vocabulary as features
        if self.text_col is not None:
            if self.word_vec==True:
                # vectors = self.loadWord2Vecfeatures(data)
                # labels = np.asarray(self.word2vec_model.wv.index_to_key)
                # indices = [self.word2vec_model.wv.index_to_key.index(w) for w in labels]  # The numerical indices of those words
                # vectors = [self.word2vec_model.wv.get_vector(w) for w in indices]

                # print(indices)
                # vocab_df = pd.DataFrame(vectors,index=labels)
                vocab_df=pd.read_csv("word2vec/word2vectfmodel/word2vec_tf.csv");
                print(vocab_df.head(5))
                # vocab_df = pd.DataFrame(vectors)
                # vocab_df.to_csv("train_review_word2vec.csv")
                X = pd.concat([X, vocab_df], axis=1, join='inner')
                print(X.head(5))
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tv = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                          ngram_range=ngram_range, stop_words=stop_words)
                v_features = self.tv.fit_transform(df[self.text_col])
                vocab = self.tv.get_feature_names()
                vocab_df = pd.DataFrame(v_features.toarray(), columns=["v_%s" % (v) for v in vocab])
                X = pd.concat([X, vocab_df], axis=1, join='inner')

        outcome_type = 'categorical' if self.is_classification else 'numerical'
        if self.outcome_col in df.columns and self.v:
            print(f'outcome column ({outcome_type}): {self.outcome_col}')
        if self.v: print(f'treatment column: {self.treatment_col}')
        if self.v: print('numerical/categorical covariates: %s' % (self.feature_names))
        if self.v and self.text_col: print('text covariate: %s' % (self.text_col))
        return (df, X, Y, T)


    def cleanText(self,text, remove_stopwords=False, stemming=False, split_text=False):
        '''
        Convert a raw review to a cleaned review
        '''
        text=str(text);
        letters_only = re.sub('[^A-Za-z]+', ' ', text) # remove non-character
        words = letters_only.lower().split()  # convert to lower case
        if remove_stopwords:  # remove stopword
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        if stemming == True:  # stemming
            #         stemmer = PorterStemmer()
            stemmer = SnowballStemmer('english')
            words = [stemmer.stem(w) for w in words]
        if split_text == True:  # split text
            return (words)
        return (" ".join(words))

    def trainWord2VecModel(self,data):

        pp.text_col = "text"
        sentences = []
        data[pp.text_col] = data[pp.text_col].astype(str)
        col_one_list = data[pp.text_col].values.tolist()
        for review in col_one_list:
            sentences.append(pp.parseSent(review, pp.tokenizer));

        print("Training Word2Vec model ...\n")

        min_word_count = 5
        num_workers = 4
        context = 5
        downsampling = 1e-3
        w2v = Word2Vec(sentences, workers=num_workers, min_count=min_word_count, window=context, sample=downsampling)
        w2v.save("w2v_features_10minwordcounts_10context.model")  # save trained word2vec model
        print("Number of words in the vocabulary list : %d \n" % len(w2v.wv.index_to_key))  # 4016
        print("Show first 10 words in the vocalbulary list  vocabulary list: \n", w2v.wv.index_to_key[0:10])

    def _preprocess_column(self, df, col, is_treatment=True):
        df = df[df[col].notnull()]
        if self._check_binary(df, col): return df, True
        d = self._check_type(df, col)
        typ = d['dtype']
        if is_treatment:
            values = sorted(df[col].unique())
            df[col].replace(values, [0, 1], inplace=True)
            if self.v: print('replaced %s in column "%s" with %s' % (values, col, [0, 1]))
        else:
            if typ == 'string':
                values = sorted(df[col].unique())
                df[col].replace(values, [0, 1], inplace=True)
                if self.v: print('replaced %s in column "%s" with %s' % (values, col, [0, 1]))
        return df, self._check_binary(df, col)


    def parseSent(self,review, tokenizer,remove_stopwords=False):
        '''
        Parse text into sentences
        '''
        raw_sentences = self.tokenizer.tokenize(review)
        sentences = []
        for raw_sentence in raw_sentences:
            setence=self.cleanText(raw_sentence, remove_stopwords, split_text=True)
            sentences.append(setence);
        return sentences

    def _check_type(self, df, col):
        dtype = None
        tmp_var = df[df[col].notnull()][col]
        if is_numeric_dtype(tmp_var):
            dtype = 'numeric'
        elif is_string_dtype(tmp_var):
            dtype = 'string'
        output = {'dtype': dtype, 'nunique': tmp_var.nunique()}
        return output

    def _check_binary(self, df, col):
        return df[col].isin([0, 1]).all()

    def loadWord2Vecfeatures(self,data):
        data2 = data[self.text_col]
        clean_review2 = []
        for document in data2:
            clean_review2.append(self.cleanText(document))
        tokens2 = []
        for review in clean_review2:
            tokens2.append(review.split())
        allVectors2 = self.averageVector(tokens2, self.word2vec_model)
        return allVectors2;


    def averageVector(self,reviews, w2v):
        total = []
        lst = w2v.wv.index_to_key
        for review in reviews:
            avgVector = w2v.wv['i'] * 0
            count = 0
            empty = True
            for word in review:
                if word in lst:
                    count += 1
                    avgVector = np.add(avgVector, w2v.wv[word])
                    empty = False
            if not empty:
                avgVector = np.divide(avgVector, count)
            total.append(avgVector)
        return total

    def runProcessingData(self,path):
        data = pd.read_csv(path)
        # data = pd.read_csv("test.csv")
        temp_treatment = 'treatment'
        df = data.copy()
        # name is confound variable
        Confound = lambda \
                p: 1 if p == 'All-New Kindle E-reader - Black, 6 Glare-Free Touchscreen Display, Wi-Fi -  Includes Special Offers,,' else 0
        df['con_found'] = df['name'].apply(Confound);
        df[temp_treatment] = df['rating'].apply(treatment_from_rating);

        df["outcome"] = simulate_Y(C=df['con_found'], T=df[temp_treatment])
        pp = DataFramePreprocesser(treatment_col=temp_treatment,
                                   outcome_col="outcome",
                                   text_col="text",
                                   include_cols=[],
                                   ignore_cols=["id", "categories", "title"],
                                   verbose=1, word_vec=True)

        dft, x, y, treatment = pp.preprocess(df, training=True)
        dft.to_csv("preprocessedWord2vec.csv")
        x.to_csv("Xword2vec.csv")
        y.to_csv("Yword2vec.csv")
        treatment.to_csv("Tword2vec.csv")
        data = pd.concat([x, y, treatment], axis=1)
        # columns = ['Unnamed: 0']
        # data.drop(columns, inplace=True, axis=1)
        data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        data.to_csv("finalword2vec.csv")


def estimate_propensities(T, C):
    # estimate treatment distribution for each strata of the confound
    # directly from the data
    df = pd.DataFrame(zip(C, T), columns=['C', 'T'])
    T_levels = set(T)
    propensities = []
    for c_level in set(C):
        subset = df.loc[df.C == c_level]
        # NOTE: subset.T => transpose
        p_TgivenC = [
            float(len(subset.loc[subset['T'] == t])) / len(subset)
            for t in T_levels
        ]
        propensities.append(p_TgivenC[1])

    return propensities


# b0  makes treatment (thm?) sepearte more (i.e. give more 1's)
# b1 1, 10, 100, makes confound (buzzy/not) seperate more (drives means apart)
# gamma 0 , 1, 4, noise level
# offset moves propensities towards the middle so sigmoid can split them into some noise
def simulate_Y(C, T, b0=0.5, b1=10, gamma=0.0, offset=0.75):
    propensities = estimate_propensities(T, C)
    # propensities = [0.27, 0.7]
    out = []
    test = defaultdict(list)
    for Ci, Ti in zip(C, T):
        noise = np.random.normal(0, 1)
        y0 = b1 * (propensities[Ci] - offset)
        y1 = b0 + y0
        y = (1 - Ti) * y0 + Ti * y1 + gamma * noise  # gamma
        simulated_prob = sigmoid(y)
        y0 = sigmoid(y0)
        y1 = sigmoid(y1)
        threshold = np.random.uniform(0, 1)
        Y = int(simulated_prob > threshold)
        out.append(Y)
        test[Ci, Ti].append(Y)

    return out


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def treatment_from_rating(rating):
    return int(rating == 5.0)


if __name__ == '__main__':
    data = pd.read_csv("test.csv")
    # print(data.head())
    temp_treatment = 'treatment'
    df = data.copy()
    # name is confound variable
    Confound = lambda \
        p: 1 if p == 'All-New Kindle E-reader - Black, 6 Glare-Free Touchscreen Display, Wi-Fi -  Includes Special Offers,,' else 0
    df['con_found'] = df['name'].apply(Confound);
    df[temp_treatment] = df['rating'].apply(treatment_from_rating);

    df["outcome"] = simulate_Y(C=df['con_found'], T=df[temp_treatment])
    pp = DataFramePreprocesser(treatment_col=temp_treatment,
                               outcome_col="outcome",
                               text_col="text",
                               include_cols=[],
                               ignore_cols=["id", "categories", "title"],
                               verbose=1,word_vec=True)

    dft, x, y, treatment = pp.preprocess(df, training=True)
    dft.to_csv("word2vec/word2vectfmodel/preprocessedtfWord2vec.csv")
    x.to_csv("word2vec/word2vectfmodel/Xtfword2vec.csv")
    y.to_csv("word2vec/word2vectfmodel/Ytfword2vec.csv")
    treatment.to_csv("word2vec/word2vectfmodel/Ttfword2vec.csv")
    data=pd.concat([x, y,treatment], axis=1)
    # columns = ['Unnamed: 0']
    # data.drop(columns, inplace=True, axis=1)
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    data.to_csv("word2vec/word2vectfmodel/finaltfword2vec.csv")
import string
import io

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
import re
import nltk
import sys
from collections import OrderedDict

from keras.layers import Flatten, Dense, Dot
from nltk import SnowballStemmer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tqdm
from tensorflow.python.ops.numpy_ops import np_config

from DataFramePreprocessing import DataFramePreprocesser


class Word2Vec(keras.Model):
  def __init__(self, vocab_size, embedding_dim, *args, **kwargs):
    super(Word2Vec, self).__init__()
    # super().__init__(*args, **kwargs)
    self.num_ns = 4
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=self.num_ns+1)
    self.dots = Dot(axes=(embedding_dim,embedding_dim))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    # we = tf.math.reduce_sum(self.target_embedding(target), axis =1)
    we = self.target_embedding(target)
    # we=we.reshape(vocab_size, self.num_ns+1,1)
    # ce = self.context_embedding(context)
    # dots = self.dots([ce, we])
    # return self.flatten(dots)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

  def custom_loss(self,x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

np_config.enable_numpy_behavior();

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')
# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def prepare_dictionary(data):
    idx = 0
    word2idx = {}
    idx2word = {}
    for sentence in data:
        for line in sentence:
            for word in line:
                if word not in word2idx.keys():
                    word2idx[word] = idx
                    idx2word[idx] = word
                    idx += 1
    vocab_size = len(word2idx.keys())
    return vocab_size, word2idx, idx2word

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=SEED,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels
# # Define the vocabulary size and the number of words in a sequence.

def cleanText(text, remove_stopwords=False, stemming=False, split_text=False):
    '''
    Convert a raw review to a cleaned review
    '''
    text = str(text);
    letters_only = re.sub('[^A-Za-z]+', ' ', text)  # remove non-character
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

def parseSent(review, tokenizer,remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review)
    sentences = []
    for raw_sentence in raw_sentences:
        setence=cleanText(raw_sentence, remove_stopwords, split_text=True)
        sentences.append(setence);
    return sentences

stopwords = nltk.corpus.stopwords.words("english")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
data = pd.read_csv("test.csv")
data['text'] = data['text'].astype(str)
data['text'].to_csv("reviews.txt",header=None, index=None, sep='\t', mode='a')
col_one_list = data['text'].values.tolist()
sentences=[]
for review in col_one_list:
    sentences.append(parseSent(review,tokenizer))
vocab_size, word2idx, idx2word = prepare_dictionary(sentences)
# vocab_size = 4096
sequence_length = 20
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

text_ds = tf.data.TextLineDataset("reviews.txt").filter(lambda x: tf.cast(tf.strings.length(x), bool))
vectorize_layer.adapt(text_ds.batch(1024))
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

# embedding_dim = 128
embedding_dim = 32
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
word2vec.fit(dataset, epochs=20)
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()


out_v = io.open('vectors.csv', 'w', encoding='utf-8')
out_m = io.open('metadata.csv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
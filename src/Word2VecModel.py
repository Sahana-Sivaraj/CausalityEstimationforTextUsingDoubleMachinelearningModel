import re

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import nltk
import pandas as pd
from nltk import SnowballStemmer


class Word2Vec:
    def __init__(self, embedding_dim=64, optimizer='sgd', epochs=10000):
        self.stopwords = nltk.corpus.stopwords.words("english")
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        data = pd.read_csv("test.csv")
        data['text'] = data['text'].astype(str)
        data['text'].to_csv("reviews.txt", header=None, index=None, sep='\t'
                            , mode='a')
        col_one_list = data['text'].values.tolist()
        sentences = []
        for review in col_one_list:
            sentences.append(self.parseSent(review, tokenizer))
        self.vocab_size, self.word2idx, self.idx2word = self.prepare_dictionary(sentences)
        # self.vocab_size=4096;
        self.X, self.Y = self.prepare_dataset(sentences, self.word2idx, self.vocab_size)
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        # embeddings structure setup
        self.W1 = tf.compat.v1.Variable(tf.random.normal([self.vocab_size, self.embedding_dim]))
        self.b1 = tf.compat.v1.Variable(tf.random.normal([self.embedding_dim]))  # bias
        self.W2 = tf.compat.v1.Variable(tf.random.normal([self.embedding_dim, self.vocab_size]))
        self.b2 = tf.compat.v1.Variable(tf.random.normal([self.vocab_size]))
        # traning data setup 
        self.x_train = tf.compat.v1.placeholder(tf.float32, shape=[None, self.vocab_size])
        self.y_train = tf.compat.v1.placeholder(tf.float32, shape=[None, self.vocab_size])
        # layers , cross_entropy_loss and optimizer setup
        self.hidden_layer = tf.add(tf.matmul(self.x_train, self.W1), self.b1)
        self.output_layer = tf.nn.softmax(tf.add(tf.matmul(self.hidden_layer, self.W2), self.b2))

        self.cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(self.y_train * tf.compat.v1.math.log(self.output_layer),
                                                                axis=1))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
        self.train_op = self.optimizer.minimize(self.cross_entropy_loss)
        self.saver = tf.compat.v1.train.Saver()

    def train(self):

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.33)
        self.init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)
        print("start w2v training")
        for i in range(self.epochs):
            itr, dataX = self.batch_dataset(self.X, self.Y)
            print(str(i)+"epoch")
            self.sess.run(itr)
            loss_val = []
            while True:
                try:
                    bx, by = self.sess.run(dataX)
                    l = self.sess.run(self.cross_entropy_loss,
                                         feed_dict={self.x_train: bx, self.y_train: by})
                    loss_val.append(l)
                except tf.errors.OutOfRangeError as e:
                    break
            if (i + 1) % 10 == 0:
                loss_mean_val = np.mean(loss_val)
                print("Epoch: {0} - Loss: {1}".format(i + 1, loss_mean_val))
        self.saver.save(self.sess, save_path="weights/word2vec.ckpt")
        print("completed w2v training")

    def save(self):
        vectors = self.sess.run(self.W1 + self.b1)
        words = list(self.word2idx.keys())
        vocab_df = pd.DataFrame(vectors)
        vocab_df.to_csv("word2vec_tf.csv")

    def cleanText(self, text, remove_stopwords=False, stemming=False, split_text=False):
        '''
          Convert a raw review to a cleaned review
          '''
        text = str(text)
        letters_only = re.sub('[^A-Za-z]+', ' ', text)  # remove non-character
        words = letters_only.lower().split()  # convert to lower case
        if remove_stopwords:  # remove stopword
            stops = set(self.stopwords.words("english"))
            words = [w for w in words if not w in stops]
        if stemming == True:  # stemming
            #         stemmer = PorterStemmer()
            stemmer = SnowballStemmer('english')
            words = [stemmer.stem(w) for w in words]
        if split_text == True:  # split text
            return (words)
        return (" ".join(words))

    def parseSent(self, review, tokenizer, remove_stopwords=False):
        raw_sentences = tokenizer.tokenize(review)
        sentences = []
        for raw_sentence in raw_sentences:
            setence = self.cleanText(raw_sentence, remove_stopwords, split_text=True)
            sentences.append(setence)
        return sentences

    def prepare_dictionary(self, data):
        # extract word2index, vocab_size and idex2word mapping for traning
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

    def prepare_dataset(self, data, word2idx, vocab_size, window=5):
        # data preparation method model traning
        X = []
        Y = []
        for sentence in data:
            for line in sentence:
                fn = window // 2
                line_len = len(line)
                if line_len > window:
                    for i in range(line_len):
                        for j in range(window):
                            idx = i + j - fn
                            if (idx != i) and (idx >= 0 and idx < line_len):
                                x = word2idx[line[i]]
                                y = word2idx[line[idx]]
                                X.append(x)
                                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def onehot_encoding(self, x, y):
        # data encoding with one-hot technique
        X = tf.one_hot(x, self.vocab_size)
        Y = tf.one_hot(y, self.vocab_size)
        return X, Y

    def batch_dataset(self, x, y, batch_size=1024, prefetch=2):
        # dividing data into batches and apply one-hot code while traning 
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(self.onehot_encoding)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        itr = tf.compat.v1.data.make_initializable_iterator(dataset)
        return itr.initializer, itr.get_next()

        
# hidden layer: which represents word vector
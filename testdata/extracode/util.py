from collections import defaultdict
import numpy as np


def ATE_unadjusted(T, Y):
    x = defaultdict(list)
    for t, y in zip(T, Y):
        x[t].append(y)
    T0 = np.mean(x[0])
    T1 = np.mean(x[1])
    return T0 - T1

def ATE_adjusted(C, T, Y):
    x = defaultdict(list)
    for c, t, y in zip(C, T, Y):
        x[c, t].append(y)

    C0_ATE = np.mean(x[0,0]) - np.mean(x[0,1])
    C1_ATE = np.mean(x[1,0]) - np.mean(x[1,1])
    return np.mean([C0_ATE, C1_ATE])

    # word2vec_filename = 'train_review_word2vec.csv'
    # with open(word2vec_filename, 'w+') as word2vec_file:
    #     for index, row in df.iterrows():
    #         tokens= row['tokens'];
    #         values=[pp.averageVector2(token) for token in tokens]
    #         model_vector = (np.mean(values, axis=0)).tolist()
    #         # print(model_vector)
    #         if index == 0:
    #             header = ",".join(str(ele) for ele in range(1000))
    #             word2vec_file.write(header)
    #             word2vec_file.write("\n")
    #         # Check if the line exists else it is vector of zeros
    #         if type(model_vector) is list:
    #             line1 = ",".join([str(vector_element) for vector_element in model_vector])
    #         else:
    #             line1 = ",".join([str(0) for i in range(1000)])
    #         word2vec_file.write(line1)
    #         word2vec_file.write('\n')

    # def averageVector2(self, word):
    #     total = []
    #     lst = self.word2vec_model.wv.index_to_key
    #     avgVector = self.word2vec_model.wv['i'] * 0
    #     count = 0
    #     empty = True
    #     if word in lst:
    #             count += 1
    #             avgVector = np.add(avgVector, self.word2vec_model.wv[word])
    #             empty = False
    #     if not empty:
    #         avgVector = np.divide(avgVector, count)
    #     total.append(avgVector)
    #     return total

# labels = np.asarray(self.word2vec_model.wv.index_to_key)  # The words we've selected
# indices = [self.word2vec_model.wv.index_to_key.index(w) for w in labels]  # The numerical indices of those words

# vectors = [self.word2vec_model.wv.get_vector(w) for w in indices]
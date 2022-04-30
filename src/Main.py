import pandas as pd

from src.DataFramePreprocessing import DataFramePreprocesser
from src.DMLNeuralNetwork import DMLNeuralNetwork
from src.ReviewSimulation import ReviewSimulation
from src.Word2VecModel import Word2Vec

def prepareVectorsformWord2Vec():
    w2v = Word2Vec(optimizer='adam', epochs=100)
    print("starting prepare_dictionary")
    w2v.train()
    w2v.save()


def runProcessingTfIdfData():
    data = pd.read_csv("test.csv")
        # data = pd.read_csv("test.csv")
    temp_treatment = 'treatment'
    df = data.copy()
        # name is confound variable
    Confound = lambda \
              p: 1 if p == 'All-New Kindle E-reader - Black, 6 Glare-Free Touchscreen Display, Wi-Fi -  Includes Special Offers,,' else 0
    df['con_found'] = df['name'].apply(Confound);
    reviewSimulation = ReviewSimulation(C=df['con_found'], T=df[temp_treatment])
    df[temp_treatment] = df['rating'].apply(reviewSimulation.treatment_from_rating);

    df["outcome"] = reviewSimulation.simulate_Y(C=df['con_found'])
    pp = DataFramePreprocesser(treatment_col=temp_treatment,
                                   outcome_col="outcome",
                                   text_col="text",
                                   include_cols=[],
                                   ignore_cols=["id", "categories", "title"],
                                   verbose=1, word_vec=True)

    dft, x, y, treatment = pp.preprocess(df, training=True)
    dft.to_csv("preprocessed.csv")
    x.to_csv("X.csv")
    y.to_csv("Y.csv")
    treatment.to_csv("T.csv")
    data = pd.concat([x, y, treatment], axis=1)
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    data.to_csv("final.csv")

def runProcessingGenisimWordVecData():
    data = pd.read_csv("test.csv")
        # data = pd.read_csv("test.csv")
    temp_treatment = 'treatment'
    df = data.copy()
        # name is confound variable
    Confound = lambda \
              p: 1 if p == 'All-New Kindle E-reader - Black, 6 Glare-Free Touchscreen Display, Wi-Fi -  Includes Special Offers,,' else 0
    df['con_found'] = df['name'].apply(Confound);
    reviewSimulation = ReviewSimulation(C=df['con_found'], T=df[temp_treatment])
    df[temp_treatment] = df['rating'].apply(reviewSimulation.treatment_from_rating);

    df["outcome"] = reviewSimulation.simulate_Y(C=df['con_found'])
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

def runProcessingTfWordVecDataProcess():
        data = pd.read_csv("test.csv")
        # print(data.head())
        temp_treatment = 'treatment'
        df = data.copy()
        # name is confound variable
        Confound = lambda \
                p: 1 if p == 'All-New Kindle E-reader - Black, 6 Glare-Free Touchscreen Display, Wi-Fi -  Includes Special Offers,,' else 0
        df['con_found'] = df['name'].apply(Confound);
        reviewSimilation = ReviewSimulation(C=df['con_found'], T=df[temp_treatment])
        df[temp_treatment] = df['rating'].apply(reviewSimilation.treatment_from_rating)

        df["outcome"] = ReviewSimulation.simulate_Y(df['con_found'])
        pp = DataFramePreprocesser(treatment_col=temp_treatment,
                                   outcome_col="outcome",
                                   text_col="text",
                                   include_cols=[],
                                   ignore_cols=["id", "categories", "title"],
                                   verbose=1, word_vec=True)

        dft, x, y, treatment = pp.preprocess(df, training=True)
        vocab_df = pd.read_csv("word2vec/word2vectfmodel/word2vec_tf.csv");
        print(vocab_df.head(5))
        x = pd.concat([x, vocab_df], axis=1, join='inner')
        dft.to_csv("word2vec/word2vectfmodel/preprocessedtfWord2vec.csv")
        x.to_csv("word2vec/word2vectfmodel/Xtfword2vec.csv")
        y.to_csv("word2vec/word2vectfmodel/Ytfword2vec.csv")
        treatment.to_csv("word2vec/word2vectfmodel/Ttfword2vec.csv")
        data = pd.concat([x, y, treatment], axis=1)
        data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        data.to_csv("word2vec/word2vectfmodel/finaltfword2vec.csv")

def trainWithDMLNeuralNetwork(option):
    if option==1:
        dmlObj = DMLNeuralNetwork()
        dmlObj.trainWithMLPRegressorDML1Algorithm(path='word2vec/word2vectfmodel/finaltfword2vec.csv',
                                              modelName='word2vec/word2vectfmodel/word2vectf')
        dmlObj.trainWithMLPClassifierDML1Algorithm(path='word2vec/word2vectfmodel/finaltfword2vec.csv',
                                               modelName='word2vec/word2vectfmodel/word2vectf')
    elif option==2:
        dmlObj = DMLNeuralNetwork()
        dmlObj.trainWithMLPRegressorDML1Algorithm(path='word2vec/word2vecgensimmodel/finalword2vec.csv',
                                                  modelName='word2vec/word2vecgensimmodel/word2vec')
        dmlObj.trainWithMLPClassifierDML1Algorithm(path='word2vec/word2vecgensimmodel/finalword2vec.csv',
                                                   modelName='word2vec/word2vecgensimmodel/word2vec')
    elif option==3:
        dmlObj = DMLNeuralNetwork()
        dmlObj.trainWithMLPRegressorDML1Algorithm(path='tf-idfmodel/final.csv',
                                                  modelName='tf-idfmodel/tf_idf')
        dmlObj.trainWithMLPClassifierDML1Algorithm(path='tf-idfmodel/final.csv',
                                                   modelName='tf-idfmodel/tf_idf')
if __name__ == '__main__':
    prepareVectorsformWord2Vec();
    runProcessingTfIdfData();
    runProcessingGenisimWordVecData();
    runProcessingTfWordVecDataProcess()
    trainWithDMLNeuralNetwork(option=1);

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import pickle as pkl

class DMLNeuralNetwork():
    def getNeuralNetworkNusianceEstimator(self, givenEstimator):
        nuisanceEstimator = MLPRegressor(hidden_layer_sizes=(15, 20),
                                             activation='relu',
                                             solver='adam',
                                             learning_rate='adaptive',
                                             max_iter=1000,
                                             learning_rate_init=0.01,
                                             alpha=0.01)
        return nuisanceEstimator

    def dml1(self, data, givenEstimator, predictorVariable, treatmentVariable, remainingVariables, numberOfFolds, thetaEstimator):
        dataList = np.array_split(data.sample(frac=1), numberOfFolds)
        result = []

        for ii in range(len(dataList)):

            # 2) Get nuisance estimator
            nusianceEstimatorM = self.getNeuralNetworkNusianceEstimator(givenEstimator)
            nusianceEstimatorG = self.getNeuralNetworkNusianceEstimator(givenEstimator)

            # Prepare D (treatment effect), Y (predictor variable), X (controls)
            mainData = dataList[ii]
            D_main = mainData[treatmentVariable] #df['treatment']
            Y_main = mainData[predictorVariable] #df['outcome']
            X_main = mainData[remainingVariables]

            dataList_ = dataList[:]
            dataList_.pop(ii)
            compData = pd.concat(dataList_)
            D_comp = compData[treatmentVariable]
            Y_comp = compData[predictorVariable]
            X_comp = compData[remainingVariables]

            # Compute g as a machine learning estimator, which is trained on the predictor variable
            g_comp = nusianceEstimatorG.fit(X_main, Y_main).predict(X_comp)
            g_main = nusianceEstimatorG.fit(X_comp, Y_comp).predict(X_main)

            # Compute m as a machine learning estimator, which is trained on the treatment variable
            m_comp = nusianceEstimatorM.fit(X_main, D_main).predict(X_comp)
            m_main = nusianceEstimatorM.fit(X_comp, D_comp).predict(X_main)

            # Compute V
            V_comp = D_comp - m_comp
            V_main = D_main - m_main
            # We provide two different theta estimators for computing theta
            if (thetaEstimator == "Opt1"):
                theta_comp = DMLNeuralNetwork.thetaEstimator1(Y_comp, V_comp, D_comp, g_comp)
                theta_main = DMLNeuralNetwork.thetaEstimator1(Y_main, V_main, D_main, g_main)
                result.append((theta_comp + theta_main) / 2)

            else:
                theta_comp = DMLNeuralNetwork.thetaEstimator2(Y_comp, V_comp, D_comp, g_comp)
                theta_main = DMLNeuralNetwork.thetaEstimator2(Y_main, V_main, D_main, g_main)
                result.append((theta_comp + theta_main) / 2)

            # Aggregate theta
        theta = np.mean(result)# coef caculation
        return theta

    @staticmethod
    def thetaEstimator1(Y, V, D, g):
        try:
            return np.mean((V * (Y - g))) / np.mean((V * D))
        except ZeroDivisionError:
            return 0

    @staticmethod
    def thetaEstimator2(Y, V, D, g):
        try:
            return np.mean((V * (Y - g))) / np.mean((V * V))
        except ZeroDivisionError:
            return 0

    def trainWithMLPRegressorDML1Algorithm(self,path,modelName):
        no = 10;
        # data = pd.read_csv("final.csv");
        data = pd.read_csv(path);
        results={};
        predictorVariable = "outcome"
        treatmentVariable = "treatment"
        remainingVariables = data.columns.difference([predictorVariable, treatmentVariable])
        dmlObj = DMLNeuralNetwork()
        givenEstimator = MLPRegressor().fit(data[remainingVariables], data.loc[:, predictorVariable])
        result = []
        for ii in range(no):
            print(ii);
            theta = dmlObj.dml1(data, givenEstimator, predictorVariable, treatmentVariable, remainingVariables, 5,
                                "Opt1")
            print(theta);
            result.append(theta)

        errors=np.std(result) / np.sqrt(len(data))

        print("MLPRegressor, Theta: ", result)
        print("MLPRegressor, Std: ", np.std(result))
        print("MLPRegressor, mean: ", np.mean(result))
        print("MLPRegressor, median: ", np.median(result))
        print(errors)
        # df = pd.DataFrame(result)
        results[len(results)] = {
            'theta_hat': result,
            'mean': np.mean(result, axis=0),
            'median': np.median(result, axis=0),
            'std': np.std(result)
        }
        modelPath=modelName+"_MLPRegressor.pkl";
        with open(modelPath, 'wb') as f:
                pkl.dump(results, f)
        return results

    def trainWithMLPClassifierDML1Algorithm(self,path,modelName):
        no = 10;
        # data = pd.read_csv("final.csv");
        data = pd.read_csv(path);
        results={};
        predictorVariable = "outcome"
        treatmentVariable = "treatment"
        remainingVariables = data.columns.difference([predictorVariable, treatmentVariable])
        dmlObj = DMLNeuralNetwork()
        givenEstimator = MLPClassifier(hidden_layer_sizes=(15, 20))\
            .fit(data[remainingVariables], data.loc[:, predictorVariable])
        result = []
        for ii in range(no):
            print(ii);
            theta = dmlObj.dml1(data, givenEstimator, predictorVariable, treatmentVariable, remainingVariables, 5,
                                "Opt1")
            print(theta);
            result.append(theta)
        print(result);
        print("MLPClassifier, Theta: ", result)
        print("MLPClassifier, Std: ", np.std(result))# standard error
        print("MLPClassifier, mean: ", np.mean(result))#coef
        print("MLPClassifier, median: ", np.median(result))
        # plt.hist(result, 50, facecolor='g', alpha=0.75)
        # plt.title("MLPClassifier - Amazon reviews")
        import seaborn as sns

        sns.set_theme(style="darkgrid")
        np.random.seed(0)
        import matplotlib.pyplot as plt

        x = np.random.randn(50)
        rug_array = np.array(result)

        ax = sns.distplot(x, rug=False, axlabel="casual DML estimates", label="test")
        sns.rugplot(rug_array, height=0.05, axis='x', ax=ax)

        plt.savefig(modelName+"MLPClassifier - Amazon reviews.png")
        plt.show()
        df = pd.DataFrame(result)
        results[len(results)] = {
            'theta_hat': result,
            'mean': np.mean(result, axis=0),
            'median': np.median(result, axis=0),
            'std':  np.std(result)
        }
        modelPath = modelName + "_MLPClassifier.pkl";
        with open(modelPath, 'wb') as f:
            pkl.dump(results, f)
        return results

    def trainRandomForestWithDML2(self):
        no = 10;
        data = pd.read_csv("final.csv");
        results = {};
        dmlObj = DMLNeuralNetwork()
        result = []
        for ii in range(no):
            print(ii);
            theta = dmlObj.dml2(data)
            print(theta);
            result.append(theta)
        print("RandomForestRegressor, Theta: ", result)
        print("RandomForestRegressor, Std: ", np.std(result))
        print("RandomForestRegressor, mean: ", np.mean(result))
        print("RandomForestRegressor, median: ", np.median(result))
        plt.hist(result, 50, facecolor='g', alpha=0.75)
        plt.title("RandomForestRegressor - Amazon reviews")
        plt.savefig("RandomForestRegressor - Amazon reviews.png")
        plt.show()
        df = pd.DataFrame(result)
        results[len(results)] = {
            'theta_hat': result,
            'mean': np.mean(result, axis=0),
            'median': np.median(result, axis=0),
            'std': np.std(result)
        }
        with open('MLPClassifier.pkl', 'wb') as f:
            pkl.dump(results, f)
        return results

    def dml2(self, data, numberOfFolds=5):
            thetas = []
            predictorVariable = "outcome"
            treatmentVariable = "treatment"
            remainingVariables = data.columns.difference([predictorVariable, treatmentVariable])
            tf = KFold(n_splits=len(data))
            for I_index, IC_index in tf.split(data[remainingVariables],data[treatmentVariable]):
                model_y = RandomForestRegressor()
                model_d = RandomForestRegressor()
                z=data[remainingVariables];
                y=data[predictorVariable];
                d=data[treatmentVariable];
                model_y.fit(z[I_index], y[I_index])
                model_d.fit(z[I_index], d[I_index])
                y_hat = model_y.predict(z[IC_index])
                d_hat = model_d.predict(z[IC_index])
                residuals = d[IC_index] - d_hat
                theta = np.matmul(
                    residuals, (y[IC_index] - y_hat)) / np.matmul(residuals, d[IC_index])
                thetas.append(theta)

            return np.mean(thetas)



if __name__ == '__main__':
    dmlObj = DMLNeuralNetwork()
    # dmlObj.trainWithMLPRegressorDML1Algorithm(path='word2vec/word2vectfmodel/finaltfword2vec.csv',
    #                                           modelName='word2vec/word2vectfmodel/word2vectf')
    dmlObj.trainWithMLPClassifierDML1Algorithm(path='D:\Project\code\ReviewsCasualEstimation\word2vec\word2vectfmodel\\finaltfword2vec.csv',
                                               modelName='D:\Project\code\ReviewsCasualEstimation\word2vec\word2vectfmodel\word2vectf')

    # plt.hist(result, 50,density=True, facecolor='g', alpha=0.75)
    # plt.title("MLPClassifier - Amazon reviews")
    # plt.savefig("D:\Project\code\ReviewsCasualEstimation\word2vec\word2vectfmodel\word2vectf" + "MLPClassifier - Amazon reviews.png")
    # plt.show()
    # dmlObj.trainRandomForestWithDML2();

    # load : get the data from file
    # print("tf-idf vector based results")
    # data = pkl.load(open("MLPRegressor.pkl", "rb"))
    # print("MLPRegressor, Theta: ", data[0]["theta_hat"])
    # print("MLPRegressor, Std: ", data[0]["mean"])
    # print("MLPRegressor, mean: ", data[0]["median"])
    # print("MLPRegressor, median: ", data[0]["std"])
    #
    # data2 = pkl.load(open("MLPClassifier.pkl", "rb"))
    # print("MLPClassifier, Theta: ", data2[0]["theta_hat"])
    # print("MLPClassifier, Std: ", data2[0]["mean"])
    # print("MLPClassifier, mean: ", data2[0]["median"])
    # print("MLPClassifier, median: ", data2[0]["std"])

    # print("word2vec ebedding model based results")
    # data = pkl.load(open("word2vec_MLPRegressor.pkl", "rb"))
    # print("MLPRegressor, Theta: ", data[0]["theta_hat"])
    # print("MLPRegressor, Std: ", data[0]["mean"])
    # print("MLPRegressor, mean: ", data[0]["median"])
    # print("MLPRegressor, median: ", data[0]["std"])
    #
    # data2 = pkl.load(open("word2vec_MLPClassifier.pkl", "rb"))
    # print("MLPClassifier, Theta: ", data2[0]["theta_hat"])
    # print("MLPClassifier, Std: ", data2[0]["mean"])
    # print("MLPClassifier, mean: ", data2[0]["median"])
    # print("MLPClassifier, median: ", data2[0]["std"])


    # print("tf word2vec ebedding model based results")
    # data = pkl.load(open("word2vec_MLPRegressor.pkl", "rb"))
    # print("MLPRegressor, Theta: ", data[0]["theta_hat"])
    # print("MLPRegressor, Std: ", data[0]["mean"])
    # print("MLPRegressor, mean: ", data[0]["median"])
    # print("MLPRegressor, median: ", data[0]["std"])
    #
    # data2 = pkl.load(open("word2vec_MLPClassifier.pkl", "rb"))
    # print("MLPClassifier, Theta: ", data2[0]["theta_hat"])
    # print("MLPClassifier, Std: ", data2[0]["mean"])
    # print("MLPClassifier, mean: ", data2[0]["median"])
    # print("MLPClassifier, median: ", data2[0]["std"])

import numpy as np
import pandas as pd
from pylab import *
import pickle
import tensorflow as tf
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from random import randint
from sklearn import preprocessing
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
import itertools
from itertools import product
import glob
import os.path
from os import path


def find_best(network, K):
    # For every file saved during the cross vallidation, it picks the one that returns the lowest loss in the test set
    # and returns the best parameters for the network and the corresponding loss associated with it
    # The argument "network" is 1 for the outcome mechanism and 2 for the treatment mechanism
    all_filenames = glob.glob("*network{}.csv".format(network))
    losses = dict()
    keywords = []

    for f in all_filenames:
        df = pd.read_csv(f)
        loss = np.array(df["1"])
        key = f.split("/")[-1]
        key = key[:-4]
        key = "-".join(key.split("-")[1:])
        if key not in losses:
            losses[key] = []
        losses[key].append(loss[~np.isnan(loss)][-1])

    best = list(losses.keys())[0]
    current = np.inf
    for key in losses.keys():
        if np.mean(losses[key]) < current:
            current = np.mean(losses[key])
            best = key
    f = open("K0-" + best + ".pkl", "rb")
    parameters = pickle.load(f)
    return parameters, current


def divide_data(M, k, seed):
    # The argument M is the data corresponding to matrix M in the main file and k is the number of folds
    # This splits the data into k random folds, as the nuisance parameters have to be learnt with one part of the data
    # and the ATE/ATT coefficients have to be learnt with the other part. The part indexed by "train" is used to
    # learn the nuisances parameters and the part "test" is used to learn the parameters of interest (ATE/ATT)
    # This data is used later in the neural_net function
    X_test = []
    Y_test = []
    X_train = []
    Y_train = []
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(M):
        x_train = M[train_index][:, :-1]
        y_train = M[train_index][:, -1]
        x_test = M[test_index][:, :-1]
        y_test = M[test_index][:, -1]

        X_train.append(x_train)
        Y_train.append(y_train)
        X_test.append(x_test)
        Y_test.append(y_test)

    return X_train, Y_train, X_test, Y_test


def weights_biases(perm):
    # Returns the weights given the dimensions specified in the argument
    # These weights are then used in the MLP function where they weight
    # each input
    initializer = tf.compat.v1.keras.initializers.glorot_normal()
    weights = {}
    for i in range(len(perm) - 1):

        weights["h" + str(i)] = tf.Variable(
            initializer([perm[i], perm[i + 1]]), trainable=True
        )
        weights["b" + str(i)] = tf.Variable(tf.zeros([1, perm[i + 1]]), trainable=True)

    return weights


def train(
    X_train, y_train, X_test, y_test, epoch, batchSize, optimizer, cost, x, y, sess
):
    # Trains the neural network given the train and test data and specifications
    # in the arguments
    # For every batch computes the loss and gives the overall loss in both, the
    # train set and the test set. The cost function is defined in the neural_net
    # function below.
    L = []
    L_test = []

    for e in range(epoch):
        K = []
        for k in range(len(X_test) // batchSize):

            batchX_test = X_test[k * batchSize : (k + 1) * batchSize]
            batchY_test = y_test[k * batchSize : (k + 1) * batchSize]
            K.append(sess.run(cost, feed_dict={x: batchX_test, y: batchY_test}))

        L_test.append(np.mean(K))
        permutation = np.random.permutation(len(X_train))

        for i in range(len(X_train) // batchSize):
            I = permutation[i * batchSize : (i + 1) * batchSize]
            sess.run(optimizer, feed_dict={x: X_train[I], y: y_train[I]})
            L.append(sess.run(cost, feed_dict={x: X_train[I], y: y_train[I]}))

            if i % 10 == 0:
                print("Step " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(L[-1]))

    return L, L_test


def predict(X, batchSize, x, pred, sess):
    # Gives the predictions of the output given the input X
    P = []
    print(len(X))
    for i in range(len(X) // batchSize):
        P.append(sess.run(pred, feed_dict={x: X[i * batchSize : (i + 1) * batchSize]}))
    return np.concatenate(P)


def MLP(x, weights):
    # Gives the output from the network. In each layer of the network, the input is
    # multiplied by the corresponding weight and trasformed with the ReLu non linearity.
    # It also returns the regularized l2 loss. The non linearity can be changed to
    # "leaky_relu" or "sigmoid"
    layer = tf.matmul(x, weights["h0"]) + weights["b0"]
    reg_loss = tf.nn.l2_loss(weights["h0"])

    for i in range(1, len(weights) // 2):
        layer = (
            tf.matmul(tf.nn.relu(layer), weights["h" + str(i)]) + weights["b" + str(i)]
        )
        reg_loss = reg_loss + tf.nn.l2_loss(weights["h" + str(i)])

    return tf.squeeze(layer), reg_loss


def save_data(
    q,
    nr_layers,
    perm,
    batch_size,
    lr,
    reg_constant,
    loss,
    network,
    L,
    L_test,
    y_test1,
    pred_y_test,
):
    # This function saves the data in files with the name indicating the k fold,
    # the set of parameters used, and the network (the network is 1 for the
    # outcome network or 2 for the treatment network)
    filename = (
        "K{}-Nr_Layers{}-perm{}-batch_size{}-lr{}-reg_constant{}-loss{}-network{}"
    )
    description = filename.format(
        q, nr_layers, perm, batch_size, lr, reg_constant, loss, network
    )
    # In each csv file, it saves the train and test loss, the actual values of the
    # output and the predicted ones
    df1 = pd.DataFrame({"Loss_Train": L})
    df2 = pd.DataFrame({"Loss_test": L_test})
    df3 = pd.DataFrame({"Actual_values": y_test1})
    df4 = pd.DataFrame({"Predicted_Values": pred_y_test})
    df5 = pd.DataFrame({"Description": description}, index=[0])
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True, axis=1)
    df.to_csv(description + ".csv")
    # Creates pickle files for each of the csv files.
    f = open(description + ".pkl", "wb")
    pickle.dump(
        {
            "Nr_Layers": nr_layers,
            "neurons": perm,
            "batch_sizes": batch_size,
            "lrs": lr,
            "reg_constants": reg_constant,
            "losses": loss,
        },
        f,
    )
    f.close()


def do_i_exist(q, nr_layers, perm, batch_size, lr, reg_constant, loss, network):
    # Checks if the file is already saved so that it does not repeat the training
    # for the same hyperparameters during the cross validation procedure later
    filename = (
        "K{}-Nr_Layers{}-perm{}-batch_size{}-lr{}-reg_constant{}-loss{}-network{}"
    )
    description = filename.format(
        q, nr_layers, perm, batch_size, lr, reg_constant, loss, network
    )

    file_name = description + ".pkl"
    return path.exists(file_name)


def neural_net(
    Y_max,
    Y_min,
    k,
    X_neural,
    Y_neural,
    X_theta,
    Y_theta,
    network,
    cross_validate,
    batch_sizes,
    Nr_Layers,
    neurons,
    lrs,
    reg_constants,
    losses,
):

    # The main neural network function, which given the input data and the
    # hyperparameters returns the output from both, the first and the second
    # network. This output is then to be used in the main file for the
    # computation of the ATE/ATT and their standard errors.
    # The data indexed by "neural" is used to learn the nuisance parameters
    # and the part indexed by "theta" is used to compute the ATE/ATT

    config = tf.ConfigProto(
        intra_op_parallelism_threads=20,
        inter_op_parallelism_threads=20,
        allow_soft_placement=True,
        device_count={"CPU": 20},
    )
    # Set the number of epochs
    epochs = 50
    # G0 are the predicted values of the first network (for the outcome mechanism)
    # with the treatment D set to 0
    # G1 are the predicted values of the first network (for the outcome mechanism)
    # with the treatment D set to 1
    # G are the predicted values for the first network (for the outcome mechanism)
    # without changing the original input
    # D is the treatment variable
    # Y is the outcome variable
    # M is the predicted outcome for the second netwrok (for the treatment mechanism)
    G_0 = []
    G_1 = []
    G = []
    D = []
    Y = []
    M = []
    if cross_validate:
        # Takes all possbile combinations of the hyperparameters set by the user and
        # cross validates to find the best combination
        possibilities = product(
            batch_sizes, neurons, lrs, reg_constants, losses, Nr_Layers
        )
    else:
        # Uses the best combinations of the hyperparameters after the cross validation
        possibilities = product(
            [batch_sizes], [neurons], [lrs], [reg_constants], [losses], [Nr_Layers]
        )
    for batch_size, neuron, lr, reg_constant, loss, nr_layers in possibilities:

        for q in range(k):
            perm = (neuron) * nr_layers
            # For every fold q, check if for that particular combination of hyperparameters
            # the file exists with the do_i_ exist function defined before. If it exists it
            # tries the next combination, if not it performs the training below

            if (
                do_i_exist(
                    q, nr_layers, perm, batch_size, lr, reg_constant, loss, network
                )
                and cross_validate
            ):
                continue
            x_neural, x_theta = X_neural[q], X_theta[q]
            y_theta = Y_theta[q]
            y_neural = Y_neural[q]
            X_train = x_neural
            X_test = x_theta
            y_train = y_neural
            y_test = y_theta

            if network == 2:
                # If network is 1 you use the whole input X (which includes treatment D as
                # the last column) to predict the outcome Y.
                # But if network is 2 we are dealing with the treatment mechanism, thus we
                # try to predict the treatment D which is in the last row in X. Thus we use
                # that as "y" and the rest of the variables in X as the input
                y_theta = x_theta[:, -1]
                x_theta = x_theta[:, :-1]
                y_train = X_train[:, -1]
                y_test = X_test[:, -1]
                X_train = X_train[:, :-1]
                X_test = X_test[:, :-1]

            tf.compat.v1.reset_default_graph()
            # Construct the boundaries for the piecewise constant learning rate
            boundary_a = (epochs * (len(X_train) // batch_size)) // 2
            boundary_b = boundary_a + boundary_a // 2
            boundaries = [boundary_a, boundary_b]

            n_input = np.shape(X_test)[1]
            # Create the tensorflow placeholders for the input and the output with the
            # corresponding shapes
            x = tf.placeholder("float", [batch_size, np.shape(X_test)[1]])
            y = tf.placeholder("float", [batch_size])
            # Use the function "weights_biases" defined before to generate the weights with the
            # dimensions specified to be then used in MLP function, multiplying the input and
            # giving the output and the reg_loss to be used in the cost function below to
            # penalize very big or very large weights
            weights = weights_biases((n_input,) + perm + (1,))
            output, reg_loss = MLP(x, weights)
            # Given a type of loss function defined by the user, it computes the cost accordingly
            if loss == "MSE":
                pred = output
                cost = tf.keras.losses.MSE(y, output)
            elif loss == "Cross Entropy":
                pred = tf.nn.sigmoid(output)
                cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
                )
            # Add the regularization term to the loss function, set the piecewise constant learning
            # rate using the boundaries created earlier and with these, use the adam optimizer to
            # find the weights that minimize the cost.
            cost = cost + reg_constant * reg_loss
            global_step = tf.Variable(0)
            learningRate = tf.train.piecewise_constant(global_step, boundaries, lr)
            optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(
                cost
            )

            init = tf.initialize_all_variables()

            with tf.Session(config=config) as sess:
                sess.run(init)
                # Lastly, train the network with the optimized weights and get the loss in both the
                # train and test set
                L, L_test = train(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    epochs,
                    batch_size,
                    optimizer,
                    cost,
                    x,
                    y,
                    sess,
                )

                print("Optimization finished")
                print("Mean squared error:", L_test[-1])

                pred_y_test = predict(X_test, batch_size, x, pred, sess)
                y_test1 = y_test[: len(pred_y_test)]

                if cross_validate:
                    # If this training is part of the cross validation, it saves the losses, the
                    # actual values and the predictions in the csv and pickle files as described
                    # in the save_data function
                    save_data(
                        q,
                        nr_layers,
                        perm,
                        batch_size,
                        lr,
                        reg_constant,
                        loss,
                        network,
                        L,
                        L_test,
                        y_test1,
                        pred_y_test,
                    )
                    continue
                # For each network selected, the function returnes the actual values and the predicted
                # ones. For network 1, it also returns the predictions with the input D set to 0 (G0)
                # and the output D set to 1 (G1) which is needed to construct the scores for obtaining
                # the ATE and ATT in the main file.
                if network == 1:
                    x_theta_1 = np.copy(x_theta)
                    x_theta_1[:, -1] = 1

                    x_theta_0 = np.copy(x_theta)
                    x_theta_0[:, -1] = 0

                    G.append(
                        predict(x_theta, batch_size, x, pred, sess) * (Y_max - Y_min)
                        + Y_min
                    )
                    G_0.append(
                        predict(x_theta_0, batch_size, x, pred, sess) * (Y_max - Y_min)
                        + Y_min
                    )
                    G_1.append(
                        predict(x_theta_1, batch_size, x, pred, sess) * (Y_max - Y_min)
                        + Y_min
                    )
                    D.append(x_theta[: len(G_0[-1]), -1])
                    Y.append((y_theta[: len(G_0[-1])]) * (Y_max - Y_min) + Y_min)
                else:
                    M.append(predict(x_theta, batch_size, x, pred, sess))
                    D.append(y_theta[: len(M[-1])])
    if cross_validate:
        return None
    if network == 1:
        return G, G_1, G_0, D, Y, L_test
    else:
        return M, D

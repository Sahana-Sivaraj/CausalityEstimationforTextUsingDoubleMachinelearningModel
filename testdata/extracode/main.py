import pandas as pd
from pylab import *
from os import path
from testdata.extracode import utils

####################################################################################################################

# The following procedure performs the double machine learning using neural networks and deep neural networks.
# It predicts the nuisance parameters (treament mechanism and outcome mechanism) using one part of the data, and
# uses the rest of the data to calculate the average treatment effects and average treatment of the treated
# effects (ATE and ATT).
# To reduce the sensitivity of the results to sample splitting, the procedure is repeated 100/1000 times and the
# coefficients of ATE and ATT (and their variance) are evaluated as the median of the coefficients among splits.
# Two standard errors are reported: (1) unadjusted standard errors (2) adjusted standard errors as in Chernozhukov
# et. al (2018) to incorporate the variation introduced by sample splitting.

################################### This section generate the sample ################################################
# Number of sample
nr_obs = 10000
# If using a different number of features, change the dimension of the weights W_d and W below
nr_features = 5
# Generate a random input X to be used later together with the treatment D defined below
X = np.random.rand(nr_obs, nr_features)
# Set some weights for each input. W_d is the weight of the input X on the treatment mechanism and W is the weight
# of the input X on the outcome mechanism
W_d = [[0.3], [0.2], [-0.4], [-1], [0.1]]
W = [[-0.5], [0.1], [-1.4], [1.2], [1.0]]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the treament variable. This is also part of the input in the neural network
D = sigmoid(np.matmul(X, W_d) + np.random.normal(size=(nr_obs, 1))) > 0.5

# Define the outcome variable
Y = D * 9 + np.matmul(X, W) + np.random.normal(size=(nr_obs, 1))

# The data is fed as a matrix where the last two rows correspond to the treatment (D) and the outcome (Y) respectively
M = np.hstack((X, D, Y))

# Normalize the data and keep Y_max and Y_min to rescale the output later when constructing the scores for ATE and ATT
M[:, :-1] -= M[:, :-1].min(0)
M[:, :-1] /= (M[:, :-1].max(0)) - (M[:, :-1].min(0))
Y_max = M[:, -1].max()
Y_min = M[:, -1].min()
M[:, -1] = (M[:, -1] - Y_min) / (Y_max - Y_min)


################### SET THE HYPERPARAMETERS OF THE NEURAL NETWORK THAT YOU WANT TO TEST AND CROSS VALIDATE ###########

batch_sizes = [128]
# If  set to 1 it is a shallow neural network, if set to 2 it is a deep network with 2 layers and so on
Nr_Layers = [1]
# Set the number of neurons in every layer of the network
neurons = [(8,), (64,)]
# Set the piecewise constant learning rates
lrs = [
    [0.01, 0.001, 0.0001],
    [0.001, 0.0001, 0.00001],
    [0.05, 0.005, 0.0005],
]
# Set the regularization constants to be used
reg_constants = [0, 0.0001, 0.001]
# Set the type of loss, 'MSE' is the mean squared error. Can also select 'Cross Entropy'
losses = ["MSE"]
# Number of folds. This method is based on sample splitting. If k=2, the sample is split into two random parts.
# Learning is done with one part of the data and the estimation with the remaining part. Then the roles are reversed.
# The final results of ATE/ATT are the averages of the estimates in each sample. Similarly for k>2.
k = 2
# To reduce the sensitivity of the results to sample splitting, the procedure is repeated 100/1000 times
nr_splits = 1000

##################################### This section does the cross validation ######################################

X_neural, Y_neural, X_theta, Y_theta = utils.divide_data(M, k, None)
# This function performs the cross validation for the first network (the output equation)
# It saves the files with information about the outcome, the predicted outcome, the train loss and the test loss
# The test loss is later used to find the best set of parameters from all the files
utils.neural_net(
    Y_max,
    Y_min,
    k,
    X_neural,
    Y_neural,
    X_theta,
    Y_theta,
    1,
    1,
    batch_sizes,
    Nr_Layers,
    neurons,
    lrs,
    reg_constants,
    losses,
)
# This function performs the cross validation for the second network (the treatment equation)
# It saves the files with information about the treatment, the predicted treatment, the train loss and the test loss
# The test loss is later used to find the best set of parameters from all the files
utils.neural_net(
    Y_max,
    Y_min,
    k,
    X_neural,
    Y_neural,
    X_theta,
    Y_theta,
    2,
    1,
    batch_sizes,
    Nr_Layers,
    neurons,
    lrs,
    reg_constants,
    losses,
)


###########  This section estimates the results with the best parameters selected with cross validation  #########
# Get the best parameters from for each network and their corresponding losses. These are used later to evaluate the
# predicted nuisance parameters and the scores of ATE and ATT.
params1, loss1 = utils.find_best(1, k)
params2, loss2 = utils.find_best(2, k)

# Saves the list of coefficients for ATE (theta_s) and  ATT (theta_s_1) and their variances
if path.exists("theta_s.csv"):
    # If it has already saved some coefficients for ATE and ATT & variances start from there and repeat
    # until the max number of splits is reached (nr_splits is defined below)
    reader = np.loadtxt("theta_s.csv", delimiter=",")
    theta_s_ate = list(reader.reshape((-1,)))
    reader = np.loadtxt("sigma2_s.csv", delimiter=",")
    var_s_ate = list(reader.reshape((-1,)))

    reader = np.loadtxt("theta_s_1.csv", delimiter=",")
    theta_s_att = list(reader.reshape((-1,)))
    reader = np.loadtxt("sigma2_s_1.csv", delimiter=",")
    var_s_att = list(reader.reshape((-1,)))

    s = len(theta_s_ate) + 1
else:
    # If starting new, create empty lists where the coefficients of ATE, ATT and their variances will be saved
    theta_s_ate = []
    var_s_ate = []
    theta_s_att = []
    var_s_att = []
    s = 1


# Redivide data and repeat the estimation s times, with nr_splits=100 or nr_splits=1000 and take the median of the
# results
while s <= nr_splits:
    X_neural, Y_neural, X_theta, Y_theta = utils.divide_data(M, k, s)
    # Get the outcome, treatment variable from network 1 and their corresponding predicitons to use later for the
    # constructions of the scores
    g, g1, g0, D_var, Y, L_test = utils.neural_net(
        Y_max,
        Y_min,
        k,
        X_neural,
        Y_neural,
        X_theta,
        Y_theta,
        network=1,
        cross_validate=0,
        **params1
    )
    # Get the outcome, treatment variable from network 2 and their corresponding predicitons to use later for the
    # constructions of the scores
    m, D_var2 = utils.neural_net(
        Y_max,
        Y_min,
        k,
        X_neural,
        Y_neural,
        X_theta,
        Y_theta,
        network=2,
        cross_validate=0,
        **params2
    )

    sigma2_ate = []
    sigma2_att = []

    m = np.array(m)
    g0 = np.array(g0)
    g1 = np.array(g1)
    g = np.array(g)
    Y = np.array(Y)
    D_var2 = np.array(D_var2)

    # Gives the ATE score which averaged over the sample gives the ATE coefficient
    phi_ate = g1 - g0 + (Y - g1) * D_var2 / m - (1 - D_var2) * (Y - g0) / (1 - m)
    # Gives the ATT  which averaged over the sample gives the ATT coefficient
    phi_att = (Y - g0) * D_var2 - m * (1 - D_var2) * (Y - g0) / (1 - m)
    pi = D_var2 + 0
    # The results are valid under the overlap assumption, i.e each observation has a positive probability of getting
    # treated and not getting treated
    # Thus, we disregard observations for which the propensity scores are too close to 1 or 0.
    phi_ate[(m < 0.01) | (m > 0.99)] = np.nan
    phi_att[(m < 0.01) | (m > 0.99)] = np.nan
    pi[(m < 0.01) | (m > 0.99)] = np.nan
    # Gives the ATE coefficient and its variance
    theta_ate = np.nanmean(phi_ate)
    var_ate = np.nanvar(phi_ate)
    # Gives the ATT coefficient and its variance
    theta_att = np.nanmean(np.nansum(phi_att, 1) / np.nansum(pi, 1))
    var_att = np.nanmean(
        ((phi_att - np.array(D_var) * theta_att) / np.nanmean(pi, 1, keepdims=True))
        ** 2
    )
    # Get the number of observations to use when calculating the standard errors. This is the number of observations
    # in the train set plus the ones in the test set
    n = len(g1[0]) + len(g1[1])

    #################################### Save all estimated ATE and their variance  #################################
    theta_s_ate.append(theta_ate)
    np.savetxt("theta_s.csv", theta_s_ate, delimiter=",")

    var_s_ate.append(var_ate)
    np.savetxt("sigma2_s.csv", var_s_ate, delimiter=",")

    #################################### Save all esitmated ATT and their variance  #################################
    theta_s_att.append(theta_att)
    np.savetxt("theta_s_1.csv", theta_s_att, delimiter=",")

    var_s_att.append(var_att)
    np.savetxt("sigma2_s_1.csv", var_s_att, delimiter=",")
    s += 1


######################################  Take the median of ATEs  ####################################################

theta_median = np.median(theta_s_ate)
# This gives the unadjusted standard error
sigma2_median_noadjustment = np.median(var_s_ate)
SE_median_noadjustment = np.sqrt(sigma2_median_noadjustment / n)
# A more robust standard error (adjusted standard error) proposed by Chernozhukov et.al (2018) is the following:
sigma2_median = np.median(var_s_ate + (theta_s_ate - theta_median) ** 2)
SE_median = np.sqrt(sigma2_median / n)


#####################################  Take the median of ATTs    ###################################################

theta_median_1 = np.median(theta_s_att)
# This gives the unadjusted standard error
sigma2_median_noadjustment_1 = np.median(var_s_att)
SE_median_noadjustment_1 = np.sqrt(sigma2_median_noadjustment_1 / n)
# A more robust standard error (adjusted standard error) proposed by Chernozhukov et.al (2018) is the following:
sigma2_median_1 = np.median(var_s_att + (theta_s_att - theta_median_1) ** 2)
SE_median_1 = np.sqrt(sigma2_median_1 / n)


###############################################  Save Results ########################################################

results = pd.DataFrame(
    {
        "Median ATE": [theta_median],
        "Median ATE SE - adjusted": [SE_median],
        "Median ATE SE - not adjusted": [SE_median_noadjustment],
        "Median ATT": [theta_median_1],
        "Median ATT SE - adjusted": [SE_median_1],
        "Median ATT SE - not adjusted": [SE_median_noadjustment_1],
    }
)

results.to_csv("Results_coefficients.csv")

import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw03_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw03_data_set_test.csv", delimiter = ",", skip_header = 1)

# get X and y values
X_train = data_set_train[:, 0:2]
y_train = data_set_train[:, 2]
X_test = data_set_test[:, 0:2]
y_test = data_set_test[:, 2]

minimum_value = -2.0
maximum_value = +2.0

def plot_figure(y, y_hat):
    fig = plt.figure(figsize = (4, 4))
    plt.axline([-12, -12], [52, 52], color = "r", linestyle = "--")
    plt.plot(y, y_hat, "k.")
    plt.xlabel("True value ($y$)")
    plt.ylabel("Predicted value ($\widehat{y}$)")
    plt.xlim([-12, 52])
    plt.ylim([-12, 52])
    plt.show()
    return(fig)


# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(X_query, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders):
    
    # your implementation starts below
    
    C = len(x1_left_borders)
    
    # 1) Compute the mean y in each cell
    cell_means = []
    for c in range(C):
        in_cell = (
            (X_train[:, 0] >= x1_left_borders[c]) & (X_train[:, 0] <  x1_right_borders[c]) &
            (X_train[:, 1] >= x2_left_borders[c]) & (X_train[:, 1] <  x2_right_borders[c])
        )
        if np.any(in_cell):
            cell_means.append(np.mean(y_train[in_cell]))
        else:
            cell_means.append(0.0)
    
    # 2) For each query point, find its cell and assign the precomputed mean
    Nq = X_query.shape[0]
    y_hat = np.zeros(Nq)
    
    for i in range(Nq):
        x1q, x2q = X_query[i, 0], X_query[i, 1]
        for c in range(C):
            if (x1_left_borders[c] <= x1q < x1_right_borders[c] and
                x2_left_borders[c] <= x2q < x2_right_borders[c]):
                y_hat[i] = cell_means[c]
                break  # stop as soon as we find the covering cell
    
    return y_hat
            
    # your implementation ends above
    
    return(y_hat)
    
bin_width = 0.50
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

x1_left_borders = np.meshgrid(left_borders, left_borders)[0].flatten()
x1_right_borders = np.meshgrid(right_borders, right_borders)[0].flatten()
x2_left_borders = np.meshgrid(left_borders, left_borders)[1].flatten()
x2_right_borders = np.meshgrid(right_borders, right_borders)[1].flatten()

y_train_hat = regressogram(X_train, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Regressogram => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("regressogram_training.pdf", bbox_inches = "tight")

y_test_hat = regressogram(X_test, X_train, y_train, x1_left_borders, x1_right_borders, x2_left_borders, x2_right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("regressogram_test.pdf", bbox_inches = "tight")



# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(X_query, X_train, y_train, bin_width):
    
    # your implementation starts below
    
    y_hat = np.zeros(X_query.shape[0])
    for i in range(X_query.shape[0]):
        x1_q, x2_q = X_query[i, 0], X_query[i, 1]

        diff1 = np.abs((X_train[:, 0] - x1_q) / bin_width)
        diff2 = np.abs((X_train[:, 1] - x2_q) / bin_width)
        weights = ((diff1 < 0.5) & (diff2 < 0.5))
        
        if np.sum(weights) > 0:
            y_hat[i] = np.sum(weights * y_train) / np.sum(weights)
        else:
            y_hat[i] = 0
            
    # your implementation ends above
    return(y_hat)

bin_width = 0.50

y_train_hat = running_mean_smoother(X_train, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Running Mean Smoother => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("running_mean_smoother_training.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(X_test, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("running_mean_smoother_test.pdf", bbox_inches = "tight")



# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(X_query, X_train, y_train, bin_width):
    
    # your implementation starts below
    
    Nq = X_query.shape[0]
    y_hat = np.zeros(Nq)
    constant = 1 / (2 * np.pi)
    bw2 = bin_width**2

    for i, x_q in enumerate(X_query):
       
        norm_sq = np.sum((X_train - x_q)**2, axis=1) / bw2

        # Gaussian kernel weights
        kernel_vals = constant * np.exp(-0.5 * norm_sq)

        # weighted average
        total_weight = kernel_vals.sum()
        if total_weight > 0:
            y_hat[i] = np.dot(kernel_vals, y_train) / total_weight
            
            
    # your implementation ends above
    return(y_hat)

bin_width = 0.08

y_train_hat = kernel_smoother(X_train, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("Kernel Smoother => RMSE on training set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_train, y_train_hat)
fig.savefig("kernel_smoother_training.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(X_test, X_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE on test set is {} when h is {}".format(rmse, bin_width))

fig = plot_figure(y_test, y_test_hat)
fig.savefig("kernel_smoother_test.pdf", bbox_inches = "tight")

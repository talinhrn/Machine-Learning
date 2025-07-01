import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",", skip_header = 1)

# get X and y values
X_train = data_set_train[:, 0:2]
y_train = data_set_train[:, 2]
X_test = data_set_test[:, 0:2]
y_test = data_set_test[:, 2]



# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}
    
    # your implementation starts below
    
    node_indices[1] = np.arange(X_train.shape[0])
    need_split[1]  = True

    while True:

        to_split = [n for n, flag in need_split.items() if flag]
        if not to_split:
            break

        for node in to_split:
            indices = node_indices[node]
            values  = y_train[indices]

            # 
            if indices.size <= P:
                is_terminal[node] = True
                need_split[node]  = False
                node_means[node]  = values.mean()
                continue

            #
            best_sse = np.inf
            best_feature, best_thresh = None, None
            for feature in (0, 1):
                col = X_train[indices, feature]
                uniq = np.unique(col)
                if uniq.size < 2:
                    continue
                candidates = (uniq[:-1] + uniq[1:]) / 2.0
                for thresh in candidates:
                    left_idx  = indices[col <= thresh]
                    right_idx = indices[col >  thresh]
                    if not left_idx.size or not right_idx.size:
                        continue
                    yL, yR = y_train[left_idx], y_train[right_idx]
                    sse = ((yL - yL.mean())**2).sum() + ((yR - yR.mean())**2).sum()
                    if sse < best_sse:
                        best_sse, best_feature, best_thresh = sse, feature, thresh

            # 
            if best_feature is None:
                is_terminal[node] = True
                need_split[node]  = False
                node_means[node]  = values.mean()
            else:
                is_terminal[node]       = False
                need_split[node]        = False
                node_features[node]     = best_feature
                node_splits[node]       = best_thresh

                left, right = 2*node, 2*node + 1
                mask = X_train[indices, best_feature] <= best_thresh
                node_indices[left]  = indices[mask]
                node_indices[right] = indices[~mask]
                need_split[left]    = True
                need_split[right]   = True
                
    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    
    num_queries = X_query.shape[0]
    y_hat = np.zeros(num_queries)

    for i in range(num_queries):
        current_node = 1

        while current_node in node_features:
            feat   = node_features[current_node]
            cutoff = node_splits[current_node]
            if X_query[i, feat] <= cutoff:
                current_node = 2 * current_node     
            else:
                current_node = 2 * current_node + 1  

        y_hat[i] = node_means[current_node]

    # your implementation ends above
    return(y_hat)



# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    
    leaf_nodes = sorted(node for node, leaf in is_terminal.items() if leaf)

    for leaf in leaf_nodes:
        path_conditions = []
        current = leaf
        while current != 1:
            parent = current // 2
            feat   = node_features[parent]
            thresh = node_splits[parent]

            if current == 2 * parent:
                op = "<="
            else:
                op = ">"
            path_conditions.insert(0, f"x{feat+1} {op} {thresh}")
            current = parent

        rule_str   = " AND ".join(path_conditions)
        prediction = node_means[leaf]
        print(f"Node {leaf:02d}: [{rule_str}] => {prediction}")
        
    # your implementation ends above

    

P = 256
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)


y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)

P_set = [2, 4, 8, 16, 32, 64, 128, 256]
rmse_train = []
rmse_test = []
for P in P_set:
    is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)

    y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
    rmse_train.append(np.sqrt(np.mean((y_train - y_train_hat)**2)))

    y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
    rmse_test.append(np.sqrt(np.mean((y_test - y_test_hat)**2)))

fig = plt.figure(figsize = (8, 4))
plt.semilogx(P_set, rmse_train, "ro-", label = "train", base = 2)
plt.semilogx(P_set, rmse_test, "bo-", label = "test", base = 2)
plt.legend()
plt.xlabel("$P$")
plt.ylabel("RMSE")
plt.show()
fig.savefig("decision_tree_P_comparison.pdf", bbox_inches = "tight")

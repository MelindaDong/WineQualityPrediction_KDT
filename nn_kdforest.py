# python nn_kdforest.py [train] [test] [random_seed] [d_list]
import sys
import numpy as np
import random
#import nn_kdtree

# set the file path
# train_file = "sample_data/train"
# test_file = "sample_data/test-sample copy"
# rand_seed = 2
# d_list_raw = '[1, 2, 3, 4]'

train_file = sys.argv[1]
test_file = sys.argv[2]
rand_seed = int(sys.argv[3])
d_list_raw = sys.argv[4]
#d_list_raw is a string, read the d_list_raw and convert it to d_list as a list
d_list = [int(num) for num in d_list_raw.strip('[]').split(',')]





# copy from nn_kdtree.py
class Node:
    def __init__(self, point, d, val, left=None, right=None):
        self.point = point
        self.d = d  # the axis
        self.val = val # the hyperplane value
        self.left = left
        self.right = right

def BuildKdTree(points, depth=0):
    if len(points) <= 0:
        return None

    # get how many dimensions
    m = len(points[0])
    axis = depth % m
    # sort the points by the axis
    points = sorted(points, key=lambda point: point[axis]) #lambda arguments: expression
    median_index = len(points) // 2
    median_point = points[median_index]
    node = Node(median_point, axis, median_point[axis])

    node.left = BuildKdTree(points[:median_index], depth + 1)
    node.right = BuildKdTree(points[median_index+1:], depth + 1)

    return node



def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_search(point, kdtree):
    best_node = None
    # initialize the best distance to be infinity
    best_dist = float("inf")
    
    def recursive_search(node):
        nonlocal best_node, best_dist
        if node is None:
            return
        dist = distance(point, node.point)
        if dist < best_dist:
            best_node = node
            best_dist = dist

        axis = node.d
        if point[axis] < node.point[axis]:
            recursive_search(node.left)
            # if the distance between the point and the hyperplane is less than the best distance
            if point[axis] - best_dist <= node.point[axis]:
                recursive_search(node.right)
        else:
            recursive_search(node.right)
            if point[axis] + best_dist >= node.point[axis]:
                recursive_search(node.left)
            
    recursive_search(kdtree)
    return best_node

# end of copy



# read the train file
train_data = np.loadtxt(train_file, delimiter=None, skiprows=1)
test_data = np.loadtxt(test_file, delimiter=None, skiprows=1)

X_train = train_data[:, :-1]
y_train = train_data[:, -1]




import random

def KdForest(data, d_list, rand_seed):
    X_train = data[:, :-1]
    n_samples, n_features = X_train.shape
    N = n_samples

    forest = []
    n_trees = len(d_list) 

    sample_size = int(N*0.8)
    # Generate sample indexes for each tree
    index_list = [i for i in range(0,N)]
    sample_indexes = []
    for j in range(0,n_trees):
        random.seed(rand_seed+j)
        subsample_idx = random.sample(index_list, sample_size)
        sample_indexes.append(subsample_idx)

    count = 0
    while count < n_trees:
        sampled_data = X_train[sample_indexes[count]]
        tree = BuildKdTree(sampled_data, d_list[count])
        forest.append(tree)
        count += 1
    return forest




from collections import Counter

def PredictKdForest(forest, data):
    labels = []
    for tree in forest:
        nn = knn_search(data, tree) # Using a 1NN search
        # get the index of the nearest node
        index = np.where((X_train == nn.point).all(axis=1))[0][0]
        nn_label = int(y_train[index])
        labels.append(nn_label)
    return Counter(labels).most_common(1)[0][0]
# Counter(labels) return a dict (counts, label)/ .most_common(1) get the most common 1 tuplie. [0][0] get the label



# Build the forest
forest = KdForest(train_data, d_list, rand_seed)

pridct_y = []
for i in range(len(test_data)):
    pridct_y.append(PredictKdForest(forest, test_data[i]))

for i in pridct_y:
    print(i)





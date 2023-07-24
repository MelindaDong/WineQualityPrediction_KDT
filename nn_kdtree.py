import sys
import numpy as np

# set the file path
# train_file = "sample_data/train"
# test_file = "sample_data/test-sample copy"
# dimension = 1

train_file = sys.argv[1]
test_file = sys.argv[2]
dimension = sys.argv[3] # in this case, 1

# read the train file
train_data = np.loadtxt(train_file, delimiter=None, skiprows=1)
test_data = np.loadtxt(test_file, delimiter=None, skiprows=1)

X_train = train_data[:, :-1]
y_train = train_data[:, -1]


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


# train the model
root = BuildKdTree(X_train)
# test the model
pridct_y = []
for i in range(len(test_data)):
    Nearest_node = (knn_search(test_data[i], root))
    # get the index of the nearest node
    index = np.where((X_train == Nearest_node.point).all(axis=1))[0][0]
    pridct_y.append(int(y_train[index]))

for i in pridct_y:
    print(i)

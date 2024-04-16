"""
Random Forest Lab

Benj McMullin
Math 403
10/24/2023
"""
from platform import uname
import os
import graphviz
from uuid import uuid4
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        
        return sample[self.column] >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    # Create a mask for the samples that match the question
    mask = data[:, question.column] >= question.value

    # Use the mask to create left and right arrays
    left = data[mask]
    right = data[~mask]

    return left, right


# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)

# Problem 2, Problem 6
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    
    best_gain = 0  # Initialize best information gain
    best_question = None  # Initialize best question
    
    num_features = data.shape[1] - 1  # Subtract 1 for the target labels column
    
    # Determine the number of features to consider
    num_subset_features = int(np.floor(np.sqrt(num_features))) if random_subset else num_features
    
    # Generate a random subset of feature indices
    feature_indices = np.random.choice(num_features, num_subset_features, replace=False)
    
    for feature in feature_indices:
        values = np.unique(data[:, feature])  # Get unique feature values
        for value in values:
            question = Question(feature, value, feature_names)  # Create a question for this feature and value

            left, right = partition(data, question)

            if num_rows(left) < min_samples_leaf or num_rows(right) < min_samples_leaf:
                continue

            gain = info_gain(data, left, right)

            if gain >= best_gain:
                best_gain = gain
                best_question = question

    if best_question is not None:
        return best_gain, best_question
    else:
        return None, None

# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if not hasattr(my_tree, "question"):#isinstance(my_tree, leaf_class):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree, filename='Digraph', leaf_class=Leaf):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv',f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)

# Prolem 4
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    # If the number of samples is smaller than twice the minimum leaf size, return a Leaf node
    if num_rows(data) < 2 * min_samples_leaf:
        return Leaf(data)

    # Find the best split for the data
    best_gain, best_question = find_best_split(data, feature_names, min_samples_leaf, random_subset)

    # If no valid question was found or if optimal gain is 0 or max depth is reached, return a Leaf node
    if best_question is None or best_gain == 0 or current_depth >= max_depth:
        return Leaf(data)

    # Split the data using the best question
    left, right = partition(data, best_question)

    # Recursively build the left and right branches
    left_branch = build_tree(left, feature_names, min_samples_leaf, max_depth, current_depth + 1, random_subset)
    right_branch = build_tree(right, feature_names, min_samples_leaf, max_depth, current_depth + 1, random_subset)

    # Return a Decision_Node with the best question and left/right branches
    return Decision_Node(best_question, left_branch, right_branch)

# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key=my_tree.prediction.get)
    elif my_tree.question.match(sample):
        return predict_tree(sample, my_tree.left)
    else:
        return predict_tree(sample, my_tree.right)

def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    correct_count = 0
    for sample in dataset:
        actual_label = sample[-1]  # Assuming labels are in the last column
        predicted_label = predict_tree(sample, my_tree)
        if actual_label == predicted_label:
            correct_count += 1
    return correct_count / len(dataset)

# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    predictions = [predict_tree(sample, tree) for tree in forest]
    return max(set(predictions), key=predictions.count)

def analyze_forest(dataset, forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    correct_count = 0
    for sample in dataset:
        actual_label = sample[-1]  # Assuming labels are in the last column
        predicted_label = predict_forest(sample, forest)
        if actual_label == predicted_label:
            correct_count += 1
    return correct_count / len(dataset)

# Problem 7
def prob7():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    # Load the data and feature names
    data = np.genfromtxt('parkinsons.csv', delimiter=',', skip_header=1)
    feature_names = np.genfromtxt('parkinsons_features.csv', delimiter=',', dtype=str)
    
    # Remove the first column from the dataset (participant ID)
    data = data[:, 1:]
    feature_names = feature_names[1:]
    
    # Randomly select 130 samples; use 100 for training and 30 for testing
    np.random.shuffle(data)
    train_data = data[:100]
    test_data = data[100:130]
    
    # Create a list to store trees in the forest
    my_forest = []
    
    # Train 5 trees for your random forest
    for _ in range(5):
        tree = build_tree(train_data, feature_names, min_samples_leaf=15, max_depth=4)
        my_forest.append(tree)
    
    # Train and analyze your forest
    start_time = time.time()
    my_accuracy = analyze_forest(test_data, my_forest)
    my_time = time.time() - start_time
    
    # Train and analyze Scikit-Learn's forest
    start_time = time.time()
    sk_forest = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
    sk_forest.fit(train_data[:, :-1], train_data[:, -1])
    sk_accuracy = sk_forest.score(test_data[:, :-1], test_data[:, -1])
    sk_time = time.time() - start_time
    
    # Use the entire dataset with an 80-20 train-test split
    np.random.shuffle(data)
    split_index = int(len(data) * 0.8)
    train_data_full = data[:split_index]
    test_data_full = data[split_index:]
    
    # Train and analyze Scikit-Learn's forest with default parameters
    start_time = time.time()
    sk_forest_full = RandomForestClassifier(n_estimators=5)
    sk_forest_full.fit(train_data_full[:, :-1], train_data_full[:, -1])
    sk_accuracy_full = sk_forest_full.score(test_data_full[:, :-1], test_data_full[:, -1])
    sk_time_full = time.time() - start_time
    
    # Return the results in three tuples
    tuple1 = (my_accuracy, my_time)
    tuple2 = (sk_accuracy, sk_time)
    tuple3 = (sk_accuracy_full, sk_time_full)
    
    return tuple1, tuple2, tuple3



if __name__ == "__main__":

    # animals = np.loadtxt('animals.csv', delimiter=',')
    # features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str,comments=None)
    # names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)
    # my_tree = build_tree(animals, features)
    
    # # Problem 1
    # question = Question(column=1, value=3, feature_names=features)
    # left, right = partition(animals, question)
    # print(len(left), len(right))

    # question = Question(column=1, value=75, feature_names=features)
    # left, right = partition(animals, question)
    # print(len(left), len(right))


    # # Problem 2
    # print(find_best_split(animals, features))

    # # Problem 4
    # draw_tree(my_tree)

    # # Problem 5
    # print(predict_tree(animals[0], my_tree))

    # # Problem 6
    # print(analyze_tree(animals, my_tree))

    # # Problem 7
    # print(prob7())

    pass

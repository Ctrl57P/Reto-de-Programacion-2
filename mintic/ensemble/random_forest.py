### Prueba 1 2 3
import numpy as np

def bootstrap_sample(X, y, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    return X[indices], y[indices]

def _entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def _information_gain(X_col, y):
    parent_entropy = _entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    
    weighted_entropy = np.sum(
        [(counts[i] / len(y)) * _entropy(y[X_col == values[i]]) for i in range(len(values))]
    )
    
    return parent_entropy - weighted_entropy

def build_id3_tree(X, y, available_features=None):
    if available_features is None:
        available_features = list(range(X.shape[1]))

    unique_classes, counts = np.unique(y, return_counts=True)
    
    if len(unique_classes) == 1:
        return unique_classes[0]
        
    if len(available_features) == 0:
        return unique_classes[np.argmax(counts)]

    gains = [_information_gain(X[:, i], y) for i in available_features]
    best_feature_idx = available_features[np.argmax(gains)]

    tree = {best_feature_idx: {}}
    remaining_features = [f for f in available_features if f != best_feature_idx]

    for value in np.unique(X[:, best_feature_idx]):
        mask = X[:, best_feature_idx] == value
        if np.sum(mask) == 0:
            tree[best_feature_idx][value] = unique_classes[np.argmax(counts)]
        else:
            tree[best_feature_idx][value] = build_id3_tree(X[mask], y[mask], remaining_features)

    return tree

def build_random_forest(X, y, n_trees=10, random_state=None):
    forest = []
    for i in range(n_trees):
        seed = random_state + i if random_state is not None else None
        X_sample, y_sample = bootstrap_sample(X, y, random_state=seed)
        tree = build_id3_tree(X_sample, y_sample)
        forest.append(tree)
    return forest

def _predict_single(tree, x):
    if not isinstance(tree, dict):
        return tree
    
    feature_idx = list(tree.keys())[0]
    value = x[feature_idx]
    
    if value in tree[feature_idx]:
        return _predict_single(tree[feature_idx][value], x)
    else:
        first_key = list(tree[feature_idx].keys())[0]
        return _predict_single(tree[feature_idx][first_key], x)

def predict_ensemble(trees, X):
    final_predictions = []
    
    for x in X:
        tree_preds = [_predict_single(tree, x) for tree in trees]
        unique_classes, counts = np.unique(tree_preds, return_counts=True)
        max_votes = np.max(counts)
        candidates = unique_classes[counts == max_votes]
        
        final_predictions.append(np.sort(candidates)[0])
        
    return np.array(final_predictions)

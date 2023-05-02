import math
import numpy as np
import pandas as pd


def entropy_func(c, n):
    return -(c*1.0/n)*math.log(c*1.0/n, 2)


def entropy_cal(c1, c2):
    if c1 == 0 or c2 == 0:
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)


def entropy_of_one_division(division):
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:
        n_c = sum(division == c)
        e = n_c*1.0/n * entropy_cal(sum(division == c),
                                    sum(division != c))
        s += e
    return s, n


def get_entropy(y_predict, y_real):
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(
        y_real[y_predict])
    s_false, n_false = entropy_of_one_division(
        y_real[~y_predict])
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false
    return s


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth

    def fit(self, x, y, par_node={}):
        if par_node is None:
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val': y.iloc[0]}
        elif self.depth >= self.max_depth:
            return None
        else:
            self.depth += 1

            col, cutoff, entropy = self.find_best_split_of_all(
                x, y)    # find one split given an information gain
            y_left = y[x.iloc[:, col] < cutoff]
            y_right = y[x.iloc[:, col] >= cutoff]
            par_node = {'col': x.columns[col], 'index_col': col,
                        'cutoff': cutoff,
                        'val': np.round(np.mean(y))}
            par_node['left'] = self.fit(
                x[x.iloc[:, col] < cutoff], y_left, {})
            par_node['right'] = self.fit(
                x[x.iloc[:, col] >= cutoff], y_right, {})
            self.trees = par_node
            return par_node

    # all features versus values, get best
    def find_best_split_of_all(self, x, y):
        # print(x.shape, y.shape)
        col = None
        min_entropy = 1
        cutoff = None
        # x.T es la transpuesta de x (x.T analiza las columnas)
        for i, c in enumerate(x.columns):
            # x.columns son las columnas
            entropy, cur_cutoff = self.find_best_split(x[c], y)
            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy

    # one feature versus values
    def find_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value  # get which ones are less than
            my_entropy = get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff

    def all_same(self, items):
        return all(x == items.iloc[0] for x in items)

    def predict(self, x):
        tree = self.trees
        x = x.reset_index(drop=True)
        results = np.array([0]*x.shape[0])

        for i, row in x.iterrows():
            rowList = list(row)  # Convertir filas en listas
            results[i] = self._get_prediction(rowList)
        return results

    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer is not None and cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']

        if cur_layer is not None:
            return cur_layer.get('val')
        else:
            return 0

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return float(sum(y_pred == y_test)) / float(len(y_test))

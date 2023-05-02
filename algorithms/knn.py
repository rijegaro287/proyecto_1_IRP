import numpy as np


class KNearestNeighbors():
    def __init__(self, X_train, y_train, n_neighbors=5, n_classes=3, weights='uniform'):
        self.X_train = X_train
        self.y_train = y_train

        self.n_neighbors = n_neighbors
        self.n_classes = n_classes
        
        self.weights = weights

    def euclidian_distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2, axis=1))

    def kneighbors(self, X_test, return_distance=False):
        dist = []
        neigh_ind = []

        point_dist = []
        for x_test in X_test:
            point_dist += [self.euclidian_distance(x_test, self.X_train)]

        for row in point_dist:
            enum_neigh = enumerate(row)
            sorted_neigh = sorted(enum_neigh,
                                  key=lambda x: x[1])[:self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]

            dist.append(dist_list)
            neigh_ind.append(ind_list)

        if return_distance:
            return np.array(dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    def predict(self, X_test):
        if self.weights == 'uniform':
            neighbors = self.kneighbors(X_test)
            y_pred = np.array([
                np.argmax(np.bincount(self.y_train[neighbor]))
                for neighbor in neighbors
            ])
            return y_pred

        if self.weights == 'distance':
            distances, neighbors = self.kneighbors(
                X_test=X_test, return_distance=True)
            y_pred = []
            for i in range(len(X_test)):
                # Los k vecinos más cercanos
                neighbor = neighbors[i]
                # Distancias a los k vecinos más cercanos
                distance = distances[i]

                # Etiquetas de los k vecinos más cercanos
                neighbors_tag = self.y_train[neighbor]
                # Cantidad de vecinos por clase
                class_count = np.bincount(
                    neighbors_tag, minlength=self.n_classes)

                # Promedio de distancias a los vecinos de cada clase
                weights = []
                for j in range(self.n_classes):
                    # Vecinos de una clase dada
                    class_neighbors = np.where(neighbors_tag == j)
                    if len(class_neighbors[0]) > 0:
                        # Si hay vecinos de la clase j, calcula el promedio de sus distancias
                        class_weight = np.sum(
                            distance[class_neighbors]) / class_count[j]
                    else:
                        # Si no hay vecinos de la clase j, asigna un peso infinito
                        class_weight = float('inf')

                    # Guarda el peso de la clase j en el índice j
                    weights.append(class_weight)

                # Selecciona la clase con el menor peso
                y_pred.append(np.argmin(weights))

            return np.array(y_pred)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return float(sum(y_pred == y_test)) / float(len(y_test))

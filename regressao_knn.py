import numpy as np

class RegressaoKnn:
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, X_train, y_train):
        self.x = np.array(X_train)
        self.y = np.array(y_train)
    
    def predict_amostra(self, x0):
        new_k = min(self.k, len(self.x))        
        d = []
        for idx, xa in enumerate(self.x):
            # Distância euclidiana
            dist = np.linalg.norm(x0-xa)
            d.append((dist, idx))
        d = sorted(d)
        # media
        soma = 0
        for idx in range(new_k):
            pos = d[idx][1]
            ya = self.y[pos]
            soma += ya
        y0 = soma / new_k
        
        return y0
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        y_predict = []
        for x0 in X_test:
            y_predict.append(self.predict_amostra(x0))
        return y_predict



# obj_ruv = RegressaoKnn(n_neighbors=1)

# x = [[1, 3],
#      [3, 5],
#      [5, 7],
#      [6, 6],
#      [8, 8],
#      [9, 9],
#      [10, 13]]

# y = [2, 3, 5, 7, 7, 8, 10]

# x = np.array(x)

# obj_ruv.fit(X_train = x, y_train = y)

# X_test = [[1, 3],
#          [3, 5],
#          [5, 7],
#          [6, 6],
#          [8, 8],
#          [9, 9],
#          [10, 13]]

# X_test = np.array(X_test)

# y_predict = obj_ruv.predict(X_test)

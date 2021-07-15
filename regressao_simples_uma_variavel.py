import numpy as np
from matplotlib import pyplot as plt

class RegressaoUmaVariavel:
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, X_train, y_train):
        self.x = X_train
        self.y = y_train
    
    def predict_escalar(self, x0):
        new_k = min(self.k, len(self.x))        
        d = []
        for idx, xa in enumerate(self.x):
            dist = (xa-x0)**2
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
        y_predict = []
        for x0 in X_test:
            y_predict.append(self.predict_escalar(x0))
        
        return y_predict

#%% 

obj_ruv = RegressaoUmaVariavel(n_neighbors=50)

x = [1, 3, 4, 6, 8, 9, 11]
y = [2, 3, 5, 7, 7, 8, 10]

obj_ruv.fit(X_train = x, y_train = y)

X_test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

y_predict = obj_ruv.predict(X_test)

plt.plot(x, y)
plt.plot(X_test, y_predict)
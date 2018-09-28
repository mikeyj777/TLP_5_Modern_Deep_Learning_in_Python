import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import pandas as pd
from datetime import datetime
from ann import ANN
from logistic import LogisticModel

# MNIST DATASET
# detection of handwritten numbers

# 784 pixels (0-783)
# first col: label - don't need for this class

t0 = datetime.now()
print("curr time:  ", t0)

df = pd.read_csv('data/mnist.csv')
print("time to load:  ", datetime.now() - t0)
M = df.as_matrix()

Y = M[:, 0]
X = M[:, 1:]

# M = 5
lr_mag = 8
reg_mag = 2
num_epochs = 100000

model = LogisticModel()
model.fit(X, Y, show_fig=False, learning_rate=10**-
          lr_mag, reg=10**-reg_mag, epochs=num_epochs)

Ypred = model.predict(X)

# inval = input("pak")

# digpred = []
#
# for i in range(X.shape[0]):
#     digpred.append("dig:  ", M[i, 0], ".  pred:  ", Ypred[i])
#
# digpred = np.asarray(digpred)

digpred = np.concatenate([[M[:, 0], Ypred]]).T

np.savetxt('data/digpred.txt', digpred, delimiter=',')

print(model.score(X, Y))

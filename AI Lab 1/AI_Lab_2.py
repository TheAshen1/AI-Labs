import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

Xa1 = np.random.normal(-1, 0.5, 100)
Xa2 = np.random.normal(1, 0.3, 100)

Xb1 = np.random.normal(1, 0.4, 100)
Xb2 = np.random.normal(-1, 0.5, 100)

plt.scatter(Xa1, Xa2, color='red', marker='o', label='A')
plt.scatter(Xb1, Xb2, color='blue', marker='x', label='B')

plt.show()

y1 = (np.zeros(100) - 1)
y2 = (np.ones(100))
Y = np.concatenate([y1, y2])

Xa = np.vstack((Xa1, Xa2))
Xb = np.vstack((Xb1, Xb2))

X = np.concatenate([Xa, Xb], 1).T

print(X)

perceptron = Perceptron(eta=0.2, numberOfIterations=5)
perceptron.fit(X, Y)
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

perceptron.predict(X)




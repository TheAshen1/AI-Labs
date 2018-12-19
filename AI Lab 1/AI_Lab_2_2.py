
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

#variant 13
Xa1 = np.array([0.5, 0, -0.5])
Xa2 = np.array([0, 1, -0.5])

Xb1 = np.array([-1, -0.5])
Xb2 = np.array([0, 1])

plt.scatter(Xa1, Xa2, color='red', marker='o', label='A')
plt.scatter(Xb1, Xb2, color='blue', marker='x', label='B')

plt.show()

y1 = np.array([-1,-1,-1])
y2 = np.array([1,1])
Y = np.concatenate([y1, y2])

Xa = np.vstack((Xa1, Xa2))
Xb = np.vstack((Xb1, Xb2))

X = np.concatenate([Xa, Xb], 1).T

print(X)

perceptron = Perceptron(eta=0.1, numberOfIterations=5)
perceptron.fit(X, Y)
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

perceptron.predict(X)




from nn import NeuralNetwork
import numpy as np

x = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
y = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
nn = NeuralNetwork((3, 3))


def train(nn):
    for i in range(100):
        nn.train_batch(x, y, 0.2)


# train(nn)
# nn.save("nn.npz")

nn.load("nn.npz")
print("{0}".format(nn.eval(x, y)))


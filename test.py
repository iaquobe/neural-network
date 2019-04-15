from nn import NeuralNetwork
import numpy as np

x = np.transpose(np.array([0, 1]))
y = np.transpose(np.array([1, 0]))
nn = NeuralNetwork((2, 4, 2))
res, _, _ = nn.predict(x)
print("\nvor training")
print("[0,1] -> {0}".format(res))
for i in range(10000):
    _, t = nn.eval(x, y)
    nn.apply_grad(.05)
res, _, _ = nn.predict(x)
print("nach training")
print("[0,1] -> {0}".format(res))

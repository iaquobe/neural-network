from nn import NeuralNetwork
import numpy as np
import mnist
import matplotlib.pyplot as plt

'''
x = np.array([np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0])])
y = np.array([np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])])

nn = NeuralNetwork((4, 4))
for i in range(10000):
    print(nn.train_batch(x, y, 0.05, 4))
print(nn.eval(x, y, 4))
'''

images = mnist.train_images()
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]) / 255
labels = mnist.train_labels()
labels_oh = np.array([np.zeros(10) for a in labels])
for oh, idx in zip(labels_oh, labels):
    np.put(oh, idx, 1)

nn = NeuralNetwork((images.shape[1], 64, 32, 10))
plt.ylabel("cost")
for i in range(10000):
    cost = nn.train_batch(images, labels_oh, .2, 100)
    print(cost)
    plt.scatter(i, cost)
plt.show()

test_images = mnist.test_images()
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2]) / 255
test_labels = mnist.test_labels()
test_labels_oh = np.array([np.zeros(10) for a in test_labels])
for oh, idx in zip(test_labels_oh, test_labels):
    np.put(oh, idx, 1)

print("network guessing correct digit {0}% of the time".format(nn.eval(test_images, test_labels_oh, 1000) * 100))
# '''

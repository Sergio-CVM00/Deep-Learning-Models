from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0, 0,], [0, 1], [1, 0], [1, 1]]
## XOR labels score is lower, because data is not linearly separable.
# AND labels = [    0,      0,      0,      1] = score: 1.0
# XOR labels = [    0,      1,      1,      0] = score: 0.5
# OR  labels = [    0,      1,      1,      1] = score: 1.0
labels = [    0,      0,      0,      1]

plt.scatter([point[0] for point in data],
            [point[1] for point in data],
            c = labels)


# Building the Perceptron
classifier = Perceptron(max_iter = 40, random_state = 22)
classifier.fit(data, labels)

# Printing the score to see if the algorithm learned AND
print(classifier.score(data, labels))

# Visualizing the Perceptron: decision boundary
# .decision_function() method:  Given a list of points, this method returns the distance those points
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

# list of points
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

# find every possible combination of those x and y values
point_grid = list(product(x_values, y_values))

# calculate distances
distances = classifier.decision_function(point_grid)

# distances stores positive and negative values. We only care about how far away a point is from the boundary
abs_distances = abs(distances)

# Matplotlib’s pcolormesh() function needs a two dimensional list
# abs_distances is a list of 10000 numbers
distances_matrix = np.reshape(abs_distances, (100,100))

# Time to draw the heat map!
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)

# legend on the heat map
plt.colorbar(heatmap)

plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Matrices A_k
A1 = np.array([[0.0, 0.0],
               [0.0, 0.16]])
A2 = np.array([[0.2,  -0.26],
               [0.23,  0.22]])
A3 = np.array([[-0.15, 0.28],
               [ 0.26, 0.24]])
A4 = np.array([[0.85,  0.04],
               [-0.04, 0.85]])

# Vectors b_k
b1 = np.array([0.0, 0.0])
b2 = np.array([0.0, 0.16])
b3 = np.array([0.0, 0.44])
b4 = np.array([0.0, 1.6])

As = [A1, A2, A3, A4]
bs = [b1, b2, b3, b4]

# Probabilities for theta
probs = [0.01, 0.07, 0.07, 0.85]

# Number of steps and initial state
T = 50000
x = np.zeros(2)  # X_0 = (0,0)
points = np.zeros((T, 2))

for t in range(T):
    theta = np.random.choice(4, p=probs)  # returns 0,1,2,3
    A = As[theta]
    b = bs[theta]
    x = A @ x + b
    points[t] = x

plt.figure(figsize=(6, 6))
plt.scatter(points[:,0], points[:,1], s=0.1, color='black')
plt.axis('equal')
plt.axis('off')
plt.title("Markov chain on R^2 starting at origin")
plt.show()
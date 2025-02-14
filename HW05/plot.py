import numpy as np
from matplotlib import pyplot as plt

threads1 = np.array([1, 2, 4, 6, 8, 10, 14, 16, 20])
times1 = np.loadtxt('problem1.out')
problem1regression = np.polyfit(threads1, times1, 2)
xspace = np.linspace(1.0, 20.0, num=100)
regression_time = np.polyval(problem1regression, xspace)
fig1 = plt.figure(1)
plt.plot(threads1, times1, marker="o", color="black",label='Runtime')
plt.plot(xspace, regression_time, '--', color='red', label='Regression')
plt.title("Problem 1 - Scaling analysis on Computing histogram with different #threads")
plt.xlabel("Thread Size")
plt.ylabel("Runtime (ms)")
plt.legend(loc="upper right")
plt.show()

threads2 = np.linspace(1,40,40)
old_a = np.loadtxt('hw4problem1.out')
vectorized_a = np.loadtxt('problem2a.out')
fig2 = plt.figure(2)
plt.plot(threads2, old_a, marker="o", color="black",label='omp')
plt.plot(threads2, vectorized_a, marker="o", color="blue",label='Vectorized+omp')
plt.title("Problem 2a - Scaling analysis on Calculating integration with vectorization")
plt.xlabel("Thread Size")
plt.ylabel("Runtime (ms)")
plt.legend(loc="upper right")
plt.show()

old_b = np.loadtxt('hw4problem1ILP.out')
vectorized_b = np.loadtxt('problem2b.out')
fig3 = plt.figure(3)
plt.plot(threads2, old_b, marker="o", color="black",label='omp+ILP')
plt.plot(threads2, vectorized_b, marker="o", color="blue",label='Vectorized+omp+ILP')
plt.title("Problem 2b - Scaling analysis on Calculating integration with ILP")
plt.xlabel("Thread Size")
plt.ylabel("Runtime (ms)")
plt.legend(loc="upper right")
plt.show()
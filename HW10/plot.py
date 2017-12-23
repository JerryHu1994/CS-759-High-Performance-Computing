import numpy as np
from matplotlib import pyplot as plt

# problem 1
exponential = np.linspace(1,24,24)
inputSize = np.exp2(exponential)
runtime1 = np.loadtxt('problem1.out')
fig1 = plt.figure(1)
plt.plot(inputSize, runtime1, marker="o", color="black",label='GPU')
plt.title("Problem 1 - Scaling analysis on Inclusive Prefix scan")
plt.xlabel("Input Size")
plt.ylabel("Inclusive Runtime (ms)")
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.legend(loc="best")
plt.show()

# problem 2
runtime2 = np.loadtxt('problem2.out')
fig2 = plt.figure(2)
plt.plot(inputSize, runtime2, marker="o", color="black",label='GPU')
plt.title("Problem 2 - Scaling analysis on GPU Reduction Operation")
plt.xlabel("Input Size")
plt.ylabel("Inclusive Runtime (ms)")
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.legend(loc="best")
plt.show()

# problem 3
fig3 = plt.figure(3)
plt.plot(inputSize, runtime1, marker="o", color="black",label='GPU')
plt.title("Problem 3 - Scaling analysis on Inclusive Prefix scan(arbitrary size)")
plt.xlabel("Input Size")
plt.ylabel("Inclusive Runtime (ms)")
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.legend(loc="best")
plt.show()
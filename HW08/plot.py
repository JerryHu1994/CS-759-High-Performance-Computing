import numpy as np
from matplotlib import pyplot as plt

#problem 2
runTime2 = np.array([50.268768, 20.635296, 20.305984])
tileSize = np.array([8, 16, 32])
fig1 = plt.figure(1)
plt.plot(tileSize, runTime2, marker="o", color="black",label='Runtime')
plt.title("Problem 2 - Scaling analysis on Matrix Multiplication different tileSize")
plt.xlabel("Tile Size")
plt.ylabel("Runtime (ms)")
plt.ylim(0,60)
plt.legend(loc="upper right")
plt.show()



#problem 3
runTime2 = np.array([2870.325684, 947.906860, 784.962158])
tileSize = np.array([8, 16, 32])
fig1 = plt.figure(1)
plt.plot(tileSize, runTime2, marker="o", color="black",label='Runtime')
plt.title("Problem 3 - Scaling analysis on Arbitrary Size Matrix Multiplication different tileSize")
plt.xlabel("Tile Size")
plt.ylabel("Runtime (ms)")
plt.ylim(500, 3000)
plt.legend(loc="upper right")
plt.show()












































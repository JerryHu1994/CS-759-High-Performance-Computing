import numpy as np
from matplotlib import pyplot as plt

#problem 1
GPUTime = np.array([0.763296, 0.848352, 1.696704, 11.529248, 190.893799, 1258.255493])
sharedTime = np.array([0.701920, 0.698752, 1.186944, 5.719360, 73.235840, 598.568909])
sharedPinnedTime = np.array([0.868480, 0.827232, 1.025472, 2.680640, 14.965088, 136.915710])
elements = np.arange(3,9,1)
fig1 = plt.figure(1)
plt.plot(elements, np.log10(GPUTime), marker="o", color="black",label='GPU')
plt.plot(elements, np.log10(sharedTime), marker="o", color="blue",label='shared')
plt.plot(elements, np.log10(sharedPinnedTime), marker="o", color="red",label='sharedPinned')
plt.title("Problem 1 - Scaling analysis on GPU, Shared Memory and Shared Pinned Memory")
plt.xlabel("Log10(Element Size)")
plt.ylabel("Log10(Inclusive Time) (ms)")
plt.xlim(2.5,8.5)
plt.ylim(-2,5)
plt.legend(loc="upper right")
plt.show()



#problem 2
CPUTime = np.array([0.013408, 0.049056, 0.196832, 0.801088, 3.225312, 13.264832, 77.285057, 206.132156, 677.322144])
GPUTime = np.array([0.791104, 1.248448, 1.273632, 1.409920, 1.890720, 4.315424, 25.836704, 30.616159, 102.725891])
GPUSharedTime = np.array([1.192352, 1.242976, 1.245312, 1.362752, 1.770592, 3.875808, 8.834816, 23.411104, 84.893661])
elements = np.arange(4, 13, 1)
fig2 = plt.figure(2)
plt.plot(elements, np.log10(CPUTime), marker="o", color="black",label='CPU')
plt.plot(elements, np.log10(GPUTime), marker="o", color="blue",label='GPU')
plt.plot(elements, np.log10(GPUSharedTime), marker="o", color="red",label='Shared')
plt.title("Problem 2 - Scaling analysis on CPU, GPU and Shared Memory")
plt.xlabel("Log2(Element Size)")
plt.ylabel("Log10(Inclusive Time) (ms)")
plt.xlim(3.5,12.5)
plt.ylim(-3,5)
plt.legend(loc="upper right")
plt.show()












































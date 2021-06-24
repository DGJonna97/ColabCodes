import matplotlib.pyplot as plt
import numpy as np

acc = np.load('CIFAR10_1000C_1500R_5E_Loss-0_46.npy')
loss = np.load('CIFAR10_1000C_1500R_5E_acc-86.npy')

#plt.plot(acc)
plt.plot(loss)
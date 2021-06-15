import matplotlib.pyplot as plt
import numpy as np

acc = np.load('acc.npy')
loss = np.load('loss.npy')

plt.plot(acc)
plt.plot(loss)
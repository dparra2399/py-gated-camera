import numpy as np
import matplotlib.pyplot as plt

function = np.genfromtxt(
    f'/Users/davidparra/PycharmProjects/py-gated-camera/light-current.csv',
    delimiter=',')

current_mA = function[:,0][:20]
power_mW = function[:,1][:20]

plt.plot(current_mA, power_mW, marker='o')
plt.xlabel("Current (mA)")
plt.ylabel("Optical Power (mW)")
plt.title("Light-Current (L-I) Curve")
plt.grid(True)
plt.show()

import numpy as np
from matplotlib import pyplot as plt


img_array = np.load('tree_detected.jpg.npy')
plt.imshow(img_array)
plt.show()

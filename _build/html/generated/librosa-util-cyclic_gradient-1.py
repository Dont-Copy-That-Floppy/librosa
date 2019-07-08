# This example estimates the gradient of cosine (-sine) from 64
# samples using direct (aperiodic) and periodic gradient
# calculation.

import matplotlib.pyplot as plt
x = 2 * np.pi * np.linspace(0, 1, num=64, endpoint=False)
y = np.cos(x)
grad = np.gradient(y)
cyclic_grad = librosa.util.cyclic_gradient(y)
true_grad = -np.sin(x) * 2 * np.pi / len(x)
plt.plot(x, true_grad, label='True gradient', linewidth=5,
         alpha=0.35)
plt.plot(x, cyclic_grad, label='cyclic_gradient')
plt.plot(x, grad, label='np.gradient', linestyle=':')
plt.legend()
# Zoom into the first part of the sequence
plt.xlim([0, np.pi/16])
plt.ylim([-0.025, 0.025])
plt.show()

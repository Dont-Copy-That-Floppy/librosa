# Compute LP coefficients of y at order 16 on entire series

y, sr = librosa.load(librosa.util.example_audio_file(), offset=30,
                     duration=10)
librosa.lpc(y, 16)

# Compute LP coefficients, and plot LP estimate of original series

import matplotlib.pyplot as plt
import scipy
y, sr = librosa.load(librosa.util.example_audio_file(), offset=30,
                     duration=0.020)
a = librosa.lpc(y, 2)
y_hat = scipy.signal.lfilter([0] + -1*a[1:], [1], y)
plt.figure()
plt.plot(y)
plt.plot(y_hat, linestyle='--')
plt.legend(['y', 'y_hat'])
plt.title('LP Model Forward Prediction')
plt.show()

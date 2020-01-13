# For normal time

import matplotlib.pyplot as plt
times = np.arange(30)
values = np.random.randn(len(times))
plt.figure()
ax = plt.gca()
ax.plot(times, values)
ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
ax.set_xlabel('Time')
plt.show()

# Manually set the physical time unit of the x-axis to milliseconds

times = np.arange(100)
values = np.random.randn(len(times))
plt.figure()
ax = plt.gca()
ax.plot(times, values)
ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit='ms'))
ax.set_xlabel('Time (ms)')

# For lag plots

times = np.arange(60)
values = np.random.randn(len(times))
plt.figure()
ax = plt.gca()
ax.plot(times, values)
ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(lag=True))
ax.set_xlabel('Lag')

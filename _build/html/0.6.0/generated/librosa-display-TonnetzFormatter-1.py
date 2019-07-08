import matplotlib.pyplot as plt
values = np.arange(6)
plt.figure()
ax = plt.gca()
ax.plot(values)
ax.yaxis.set_major_formatter(librosa.display.TonnetzFormatter())
ax.set_ylabel('Tonnetz')

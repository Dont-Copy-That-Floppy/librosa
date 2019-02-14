import matplotlib.pyplot as plt
values = np.arange(12)
plt.figure()
ax = plt.gca()
ax.plot(values)
ax.yaxis.set_major_formatter(librosa.display.ChromaFormatter())
ax.set_ylabel('Pitch class')

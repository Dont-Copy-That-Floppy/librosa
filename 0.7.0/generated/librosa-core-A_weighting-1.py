# Get the A-weighting for CQT frequencies

import matplotlib.pyplot as plt
freqs = librosa.cqt_frequencies(108, librosa.note_to_hz('C1'))
aw = librosa.A_weighting(freqs)
plt.plot(freqs, aw)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Weighting (log10)')
plt.title('A-Weighting of CQT frequencies')
plt.show()

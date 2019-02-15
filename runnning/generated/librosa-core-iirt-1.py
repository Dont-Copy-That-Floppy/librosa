import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.util.example_audio_file())
D = np.abs(librosa.iirt(y))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='cqt_hz', x_axis='time')
plt.title('Semitone spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

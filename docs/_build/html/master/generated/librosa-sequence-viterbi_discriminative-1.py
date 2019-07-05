# This example constructs a simple, template-based discriminative chord estimator,
# using CENS chroma as input features.

# .. note:: this chord model is not accurate enough to use in practice. It is only
# intended to demonstrate how to use discriminative Viterbi decoding.

# Create templates for major, minor, and no-chord qualities
maj_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0])
min_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0])
N_template   = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 4.
# Generate the weighting matrix that maps chroma to labels
weights = np.zeros((25, 12), dtype=float)
labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
          'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
          'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
          'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
          'N']
for c in range(12):
    weights[c, :] = np.roll(maj_template, c) # c:maj
    weights[c + 12, :] = np.roll(min_template, c)  # c:min
weights[-1] = N_template  # the last row is the no-chord class
# Make a self-loop transition matrix over 25 states
trans = librosa.sequence.transition_loop(25, 0.9)

# Load in audio and make features
y, sr = librosa.load(librosa.util.example_audio_file())
chroma = librosa.feature.chroma_cens(y=y, sr=sr, bins_per_octave=36)
# Map chroma (observations) to class (state) likelihoods
probs = np.exp(weights.dot(chroma))  # P[class | chroma] proportional to exp(template' chroma)
probs /= probs.sum(axis=0, keepdims=True)  # probabilities must sum to 1 in each column
# Compute independent frame-wise estimates
chords_ind = np.argmax(probs, axis=0)
# And viterbi estimates
chords_vit = librosa.sequence.viterbi_discriminative(probs, trans)

# Plot the features and prediction map
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(weights, x_axis='chroma')
plt.yticks(np.arange(25) + 0.5, labels)
plt.ylabel('Chord')
plt.colorbar()
plt.tight_layout()
plt.show()

# And plot the results
plt.figure(figsize=(10, 4))
librosa.display.specshow(probs, x_axis='time', cmap='gray')
plt.colorbar()
times = librosa.frames_to_time(np.arange(len(chords_vit)))
plt.scatter(times, chords_ind + 0.75, color='lime', alpha=0.5, marker='+', s=15, label='Independent')
plt.scatter(times, chords_vit + 0.25, color='deeppink', alpha=0.5, marker='o', s=15, label='Viterbi')
plt.yticks(0.5 + np.unique(chords_vit), [labels[i] for i in np.unique(chords_vit)], va='center')
plt.legend()
plt.tight_layout()
plt.show()

# Load audio, detect beat frames, and subdivide in twos by CQT

y, sr = librosa.load(librosa.util.example_audio_file(), duration=8)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512))
subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
subseg_t = librosa.frames_to_time(subseg, sr=sr, hop_length=512)
subseg
# array([  0,   2,   4,  21,  23,  26,  43,  55,  63,  72,  83,
# 97, 102, 111, 122, 137, 142, 153, 162, 180, 182, 185,
# 202, 210, 221, 231, 241, 256, 261, 271, 281, 296, 301,
# 310, 320, 339, 341, 344, 361, 368, 382, 389, 401, 416,
# 420, 430, 436, 451, 456, 465, 476, 489, 496, 503, 515,
# 527, 535, 544, 553, 558, 571, 578, 590, 607, 609, 638])

import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(cqt,
                                                 ref=np.max),
                         y_axis='cqt_hz', x_axis='time')
lims = plt.gca().get_ylim()
plt.vlines(beat_times, lims[0], lims[1], color='lime', alpha=0.9,
           linewidth=2, label='Beats')
plt.vlines(subseg_t, lims[0], lims[1], color='linen', linestyle='--',
           linewidth=1.5, alpha=0.5, label='Sub-beats')
plt.legend(frameon=True, shadow=True)
plt.title('CQT + Beat and sub-beat markers')
plt.tight_layout()

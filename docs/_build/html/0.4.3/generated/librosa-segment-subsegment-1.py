# Load audio, detect beat frames, and subdivide in twos by CQT

y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
cqt = librosa.cqt(y, sr=sr, hop_length=512)
subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
subseg
# array([  0,   2,   4,  21,  23,  26,  43,  55,  63,  72,  83,
# 97, 102, 111, 122, 137, 142, 153, 162, 180, 182, 185,
# 202, 210, 221, 231, 241, 256, 261, 271, 281, 296, 301,
# 310, 320, 339, 341, 344, 361, 368, 382, 389, 401, 416,
# 420, 430, 436, 451, 456, 465, 476, 489, 496, 503, 515,
# 527, 535, 544, 553, 558, 571, 578, 590, 607, 609, 638])

import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(librosa.logamplitude(cqt**2,
                                              ref_power=np.max),
                         y_axis='cqt_hz', x_axis='time')
plt.vlines(beats, 0, cqt.shape[0], color='r', alpha=0.5,
           label='Beats')
plt.vlines(subseg, 0, cqt.shape[0], color='b', linestyle='--',
           alpha=0.5, label='Sub-beats')
plt.legend(frameon=True, shadow=True)
plt.title('CQT + Beat and sub-beat markers')
plt.tight_layout()

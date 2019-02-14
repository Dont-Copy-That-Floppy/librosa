melfb = librosa.filters.mel(22050, 2048)
melfb
# array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
# [ 0.   ,  0.   , ...,  0.   ,  0.   ],
# ...,
# [ 0.   ,  0.   , ...,  0.   ,  0.   ],
# [ 0.   ,  0.   , ...,  0.   ,  0.   ]])

# Clip the maximum frequency to 8KHz

librosa.filters.mel(22050, 2048, fmax=8000)
# array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
# [ 0.  ,  0.  , ...,  0.  ,  0.  ],
# ...,
# [ 0.  ,  0.  , ...,  0.  ,  0.  ],
# [ 0.  ,  0.  , ...,  0.  ,  0.  ]])

import matplotlib.pyplot as plt
plt.figure()
librosa.display.specshow(melfb, x_axis='linear')
plt.ylabel('Mel filter')
plt.title('Mel filter bank')
plt.colorbar()
plt.tight_layout()

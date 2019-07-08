.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_viterbi.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_viterbi.py:


================
Viterbi decoding
================

This notebook demonstrates how to use Viterbi decoding to impose temporal
smoothing on frame-wise state predictions.

Our working example will be the problem of silence/non-silence detection.



.. code-block:: python


    # Code source: Brian McFee
    # License: ISC

    ##################
    # Standard imports
    from __future__ import print_function
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa

    import librosa.display







Load an example signal



.. code-block:: python

    y, sr = librosa.load('audio/sir_duke_slow.mp3')


    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))


    ###################
    # Plot the spectrum
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.tight_layout()




.. image:: /auto_examples/images/sphx_glr_plot_viterbi_001.png
    :class: sphx-glr-single-img




As you can see, there are periods of silence and
non-silence throughout this recording.




.. code-block:: python


    # As a first step, we can plot the root-mean-square (RMS) curve
    rms = librosa.feature.rms(y=y)[0]

    times = librosa.frames_to_time(np.arange(len(rms)))

    plt.figure(figsize=(12, 4))
    plt.plot(times, rms)
    plt.axhline(0.02, color='r', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('RMS')
    plt.axis('tight')
    plt.tight_layout()

    # The red line at 0.02 indicates a reasonable threshold for silence detection.
    # However, the RMS curve occasionally dips below the threshold momentarily,
    # and we would prefer the detector to not count these brief dips as silence.
    # This is where the Viterbi algorithm comes in handy!




.. image:: /auto_examples/images/sphx_glr_plot_viterbi_002.png
    :class: sphx-glr-single-img




As a first step, we will convert the raw RMS score
into a likelihood (probability) by logistic mapping

  :math:`P[V=1 | x] = \frac{\exp(x - \tau)}{1 + \exp(x - \tau)}`

where :math:`x` denotes the RMS value and :math:`\tau=0.02` is our threshold.
The variable :math:`V` indicates whether the signal is non-silent (1) or silent (0).

We'll normalize the RMS by its standard deviation to expand the
range of the probability vector



.. code-block:: python


    r_normalized = (rms - 0.02) / np.std(rms)
    p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

    # We can plot the probability curve over time:

    plt.figure(figsize=(12, 4))
    plt.plot(times, p, label='P[V=1|x]')
    plt.axhline(0.5, color='r', alpha=0.5, label='Descision threshold')
    plt.xlabel('Time')
    plt.axis('tight')
    plt.legend()
    plt.tight_layout()




.. image:: /auto_examples/images/sphx_glr_plot_viterbi_003.png
    :class: sphx-glr-single-img




which looks much like the first plot, but with the decision threshold
shifted to 0.5.  A simple silence detector would classify each frame
independently of its neighbors, which would result in the following plot:



.. code-block:: python



    plt.figure(figsize=(12, 6))
    ax = plt.subplot(2,1,1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.subplot(2,1,2, sharex=ax)
    plt.step(times, p>=0.5, label='Non-silent')
    plt.xlabel('Time')
    plt.axis('tight')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.tight_layout()




.. image:: /auto_examples/images/sphx_glr_plot_viterbi_004.png
    :class: sphx-glr-single-img




We can do better using the Viterbi algorithm.
We'll use state 0 to indicate silent, and 1 to indicate non-silent.
We'll assume that a silent frame is equally likely to be followed
by silence or non-silence, but that non-silence is slightly
more likely to be followed by non-silence.
This is accomplished by building a self-loop transition matrix,
where `transition[i, j]` is the probability of moving from state
`i` to state `j` in the next frame.



.. code-block:: python


    transition = librosa.sequence.transition_loop(2, [0.5, 0.6])
    print(transition)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[0.5 0.5]
     [0.4 0.6]]


Our `p` variable only indicates the probability of non-silence,
so we need to also compute the probability of silence as its complement.



.. code-block:: python


    full_p = np.vstack([1 - p, p])
    print(full_p)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[0.6666718  0.66667724 0.66667974 0.66668206 0.6666705  0.66665936
      0.66665924 0.66664785 0.6666527  0.6666621  0.6666511  0.66666025
      0.66666305 0.6528213  0.55930537 0.5039653  0.46875846 0.4450308
      0.44209796 0.44649738 0.45015788 0.45296752 0.4719298  0.5008869
      0.5337621  0.57154596 0.606639   0.6330662  0.65604496 0.66306376
      0.66561913 0.666324   0.6665821  0.66669273 0.6667012  0.66669333
      0.66666317 0.6666628  0.6666759  0.66669154 0.66671777 0.6667079
      0.66669273 0.6666881  0.6666942  0.666708   0.66669214 0.66659313
      0.66656935 0.66656494 0.6665878  0.6666758  0.66668403 0.6666858
      0.66665304 0.66665214 0.6666681  0.6666596  0.66597724 0.63325614
      0.5652859  0.51315    0.46493632 0.4289655  0.4174766  0.43100977
      0.45571315 0.4427998  0.40920138 0.36801296 0.33117193 0.32734364
      0.33589643 0.34995496 0.36855036 0.38244247 0.39775676 0.40771073
      0.41529346 0.43720037 0.44843477 0.45536393 0.4675148  0.4552667
      0.45985067 0.4703753  0.49201477 0.52797276 0.5675837  0.6191
      0.6520691  0.6615268  0.6648917  0.66609764 0.6665411  0.6666868
      0.66657996 0.569123   0.46814525 0.40058404 0.34356654 0.31267142
      0.3057881  0.3367235  0.38994116 0.3435306  0.27476114 0.21300775
      0.16491199 0.15554297 0.15958053 0.15955102 0.16644597 0.17588884
      0.18820906 0.21075386 0.25177675 0.26303136 0.26994956 0.2619127
      0.22955441 0.22823441 0.2344361  0.2454806  0.27291375 0.332012
      0.40812033 0.5046606  0.6122711  0.6514727  0.6617687  0.6653024
      0.66616476 0.6663498  0.6663789  0.66624826 0.66633976 0.66631037
      0.66634905 0.66654646 0.66662276 0.6667176  0.666733   0.65877247
      0.5928712  0.5039612  0.45510358 0.4198388  0.40622747 0.42472553
      0.44260973 0.46210462 0.4737749  0.4876116  0.49827343 0.5211004
      0.5527519  0.5727309  0.5550958  0.5140325  0.4782498  0.45356625
      0.44303483 0.46430022 0.49164522 0.5032617  0.5192528  0.4985705
      0.47524446 0.4729641  0.4741633  0.4765315  0.47928607 0.4899767
      0.513384   0.5423038  0.58593965 0.6291379  0.6509727  0.66193837
      0.66451293 0.6653484  0.66545403 0.61951673 0.4705072  0.37771958
      0.32032865 0.28407103 0.2920298  0.3153656  0.3394733  0.36354804
      0.38816285 0.41107035 0.4289044  0.45955974 0.46007717 0.4506691
      0.4351393  0.40235102 0.39231306 0.39058912 0.39216697 0.42476636
      0.42307156 0.38793927 0.3500241  0.3053974  0.2909218  0.29614717
      0.3141079  0.3347456  0.3778175  0.43760216 0.507926   0.60125047
      0.65051675 0.62032306 0.42866963 0.3023131  0.22557527 0.17425424
      0.1579969  0.16310644 0.1877228  0.21032476 0.21690434 0.2107681
      0.18132776 0.16777152 0.17310858 0.18597937 0.20130664 0.21143132
      0.22859573 0.2762518  0.34608555 0.44942862 0.5088064  0.28546536
      0.17546445 0.10721451 0.07273751 0.07060146 0.07627481 0.09409767
      0.11961138 0.16860497 0.24203527 0.3360378  0.5044184  0.6321951
      0.6527614  0.6605841  0.6648702  0.6659353  0.66608727 0.43917018
      0.18581986 0.09230405 0.05653888 0.0455963  0.05672073 0.07201582
      0.10393089 0.09749305 0.08757418 0.08931983 0.07751006 0.09155339
      0.11476088 0.134727   0.1479376  0.15648991 0.14926606 0.12743425
      0.12095958 0.13677937 0.11081976 0.08474815 0.06102782 0.04243314
      0.04023486 0.04601079 0.05660105 0.07429272 0.12322688 0.208314
      0.35310978 0.5876898  0.6517385  0.65917456 0.66296625 0.6642662
      0.6655711  0.6661889  0.6663963  0.6664916  0.66649944 0.66647243
      0.66647685 0.6664698  0.6664419  0.6663797  0.66630363 0.66631365
      0.6663797  0.6663891 ]
     [0.3333282  0.33332273 0.33332023 0.33331794 0.3333295  0.33334064
      0.3333408  0.33335215 0.33334732 0.3333379  0.33334887 0.33333975
      0.33333698 0.34717867 0.44069463 0.49603468 0.53124154 0.5549692
      0.55790204 0.5535026  0.5498421  0.5470325  0.5280702  0.49911308
      0.4662379  0.42845407 0.39336094 0.36693382 0.343955   0.33693627
      0.33438084 0.33367595 0.33341792 0.33330724 0.33329883 0.33330667
      0.33333686 0.3333372  0.33332404 0.3333085  0.3332822  0.3332921
      0.3333073  0.33331192 0.33330578 0.33329204 0.33330783 0.33340687
      0.33343065 0.3334351  0.33341217 0.33332422 0.333316   0.33331418
      0.33334696 0.33334786 0.33333188 0.33334044 0.33402273 0.36674386
      0.4347141  0.48685005 0.5350637  0.5710345  0.5825234  0.56899023
      0.54428685 0.5572002  0.5907986  0.63198704 0.66882807 0.67265636
      0.66410357 0.65004504 0.63144964 0.6175575  0.60224324 0.59228927
      0.58470654 0.56279963 0.55156523 0.5446361  0.5324852  0.5447333
      0.54014933 0.5296247  0.50798523 0.47202724 0.43241632 0.38090006
      0.34793088 0.33847317 0.3351083  0.3339024  0.33345887 0.33331326
      0.33342    0.43087694 0.53185475 0.59941596 0.65643346 0.6873286
      0.6942119  0.6632765  0.61005884 0.6564694  0.72523886 0.78699225
      0.835088   0.84445703 0.8404195  0.840449   0.833554   0.82411116
      0.81179094 0.78924614 0.74822325 0.73696864 0.73005044 0.7380873
      0.7704456  0.7717656  0.7655639  0.7545194  0.72708625 0.667988
      0.59187967 0.49533936 0.38772896 0.3485273  0.33823133 0.3346976
      0.33383527 0.33365017 0.33362108 0.33375174 0.33366024 0.33368963
      0.33365092 0.33345357 0.33337727 0.3332824  0.33326694 0.34122753
      0.4071288  0.4960388  0.5448964  0.5801612  0.59377253 0.57527447
      0.5573903  0.5378954  0.5262251  0.5123884  0.50172657 0.4788996
      0.44724807 0.4272691  0.44490418 0.4859675  0.5217502  0.54643375
      0.5569652  0.5356998  0.5083548  0.4967383  0.48074722 0.5014295
      0.52475554 0.5270359  0.5258367  0.5234685  0.5207139  0.5100233
      0.48661605 0.45769623 0.41406035 0.37086216 0.34902728 0.33806163
      0.33548707 0.33465156 0.334546   0.3804833  0.5294928  0.6222804
      0.67967135 0.715929   0.7079702  0.6846344  0.6605267  0.63645196
      0.61183715 0.58892965 0.5710956  0.54044026 0.53992283 0.5493309
      0.5648607  0.597649   0.60768694 0.6094109  0.607833   0.57523364
      0.57692844 0.6120607  0.6499759  0.6946026  0.7090782  0.70385283
      0.6858921  0.6652544  0.6221825  0.56239784 0.49207398 0.39874953
      0.34948325 0.3796769  0.57133037 0.6976869  0.77442473 0.82574576
      0.8420031  0.83689356 0.8122772  0.78967524 0.78309566 0.7892319
      0.81867224 0.8322285  0.8268914  0.81402063 0.79869336 0.7885687
      0.77140427 0.7237482  0.65391445 0.5505714  0.4911936  0.71453464
      0.82453555 0.8927855  0.9272625  0.92939854 0.9237252  0.9059023
      0.8803886  0.83139503 0.75796473 0.6639622  0.49558163 0.36780486
      0.3472386  0.3394159  0.3351298  0.3340647  0.3339127  0.5608298
      0.81418014 0.90769595 0.9434611  0.9544037  0.94327927 0.9279842
      0.8960691  0.90250695 0.9124258  0.9106802  0.92248994 0.9084466
      0.8852391  0.865273   0.8520624  0.8435101  0.85073394 0.87256575
      0.8790404  0.86322063 0.88918024 0.91525185 0.9389722  0.95756686
      0.95976514 0.9539892  0.94339895 0.9257073  0.8767731  0.791686
      0.6468902  0.41231018 0.34826148 0.3408254  0.33703372 0.3357338
      0.33442888 0.3338111  0.33360368 0.33350834 0.33350056 0.33352754
      0.33352315 0.33353016 0.33355805 0.33362034 0.33369634 0.33368638
      0.3336203  0.3336109 ]]


Now, we're ready to decode!
We'll use `viterbi_discriminative` here, since the inputs are
state likelihoods conditional on data (in our case, data is rms).



.. code-block:: python


    states = librosa.sequence.viterbi_discriminative(full_p, transition)

    # sphinx_gallery_thumbnail_number = 5
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(2,1,1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr)
    plt.xlabel('')
    ax.tick_params(labelbottom=False)
    plt.subplot(2, 1, 2, sharex=ax)
    plt.step(times, p>=0.5, label='Frame-wise')
    plt.step(times, states, linestyle='--', color='orange', label='Viterbi')
    plt.xlabel('Time')
    plt.axis('tight')
    plt.ylim([0, 1.05])
    plt.legend()





.. image:: /auto_examples/images/sphx_glr_plot_viterbi_005.png
    :class: sphx-glr-single-img




Note how the Viterbi output has fewer state changes than the frame-wise
predictor, and it is less sensitive to momentary dips in energy.
This is controlled directly by the transition matrix.
A higher self-transition probability means that the decoder is less
likely to change states.


**Total running time of the script:** ( 0 minutes  1.564 seconds)


.. _sphx_glr_download_auto_examples_plot_viterbi.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_viterbi.py <plot_viterbi.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_viterbi.ipynb <plot_viterbi.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_

import numpy as np
import pandas as pd

import librosa.display
import librosa.core as librosa
import pydub
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from pydub import AudioSegment
from ClassifierPack import *

AUTOCORRELATE_AMOUNT = 0.1
PLOT_AUTOCORRELATION_QUANTILE = 0.5
CLASS_PARTS = 5

if __name__ == '__main__':

    songs = ["bach_invention_in_e.mp3", "HaveYourselfAMerryLittleChristmas.mp3",
             "mozart_sonata545.mp3", "Sia-Chandelier.mp3", "therefore_i_am_billie_eilish.mp3"]
    for i in songs:
        filename = i

        print("\n== SONG {} ==\n".format(filename))

        # Create WaveClassifier() object from .mp3 file and pre-process it
        soundObject = WaveClassifier(filename)
        soundObject.transform_audio_from_mp3(filename[:-4])

        print("Loading ...")
        soundObject.find_beat_per_second()
        print("Count auto-correlation ...")
        soundObject.autocorrelate_bps(AUTOCORRELATE_AMOUNT)

        # Plot the frequency wave
        # fig, ax = plt.subplots()
        # ax.plot(soundObject.y)
        # fig.set_size_inches(15, 5)

        # Smooth auto-correlation timeline and plot it
        print("Smooth auto-correlation ...")
        smoothed_wave = soundObject.moving_average(soundObject.autocor_bps[0],
                                                   window=int(soundObject.autocor_bps[1] / 2),
                                                   forecast=False)
        # soundObject.plot_ac(smoothed_wave, PLOT_AUTOCORRELATION_QUANTILE, int(soundObject.autocor_bps[1]))

        # Catch peaks of low and high points on auto-correlation plot. And then plot them
        print("Catch peaks ...")
        soundObject.catch_peak(20, smoothed_wave)
        soundObject.plot_peaks(smoothed_wave, soundObject.peaks)

        # Transform peaks to timestamps from original wave and find mel frequency for windows.
        print("Find mel-frequency windows ...")
        time_peaks = soundObject.transform_indexes_to_time(soundObject.y, smoothed_wave, soundObject.peaks)
        sorted_mix_peaks = mix_and_sort(time_peaks[0], time_peaks[1])
        soundObject.mel_frequency_windows(sorted_mix_peaks)
        soundObject.mel_windows_avarages()
        # soundObject.plot_mel(20)

        print("Classify wave ...")
        mixed_peaks_ac = mix_and_sort(soundObject.peaks[0], soundObject.peaks[1])
        # print(mixed_peaks_ac[17:30])
        soundObject.wave_classify(CLASS_PARTS)

        # Without zero
        print("Create dict of classes and visualise ...")
        synchronize_peaks_without_zero = sorted_mix_peaks[1::]
        autocor_peaks = soundObject.transform_timest_to_autocor(synchronize_peaks_without_zero, smoothed_wave)
        soundObject.create_dict_of_classes(CLASS_PARTS, autocor_peaks, smoothed_wave)

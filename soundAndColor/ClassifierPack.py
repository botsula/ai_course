import numpy as np
import pandas as pd

import librosa.display
import librosa
import pydub
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from pydub import AudioSegment


def mix_and_sort(arr_1, arr_2):
    mixed = np.append(arr_1, arr_2)
    return np.sort(mixed)


class WaveClassifier:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.track_info = []
        self.y = None
        self.sr = None
        self.bps = None
        self.beat_times = None
        self.autocor_bps = None
        self.peaks = None
        self.mel_windows = None
        self.mel_avarages = None
        self.classes_by_mel = None
        self.dict_of_classes = None

    def find_beat_per_second(self):
        y, sr = librosa.load(self.audio_file)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        self.track_info = [y, sr, tempo, beat_times]
        self.y = y
        self.sr = sr
        self.bps = tempo
        self.beat_times = beat_times
        return [y, sr, tempo, beat_times]

    def autocorrelate_bps(self, amount=1):
        if self.sr:
            print('sr = ', self.sr)
            # lag = bps * amount
            lag = self.bps * amount
            ac = librosa.autocorrelate(self.y, max_size=lag * self.sr / 512)
            self.autocor_bps = [ac, lag]
            return [ac, lag]
        else:
            print("[ autocorrelate_bps ] Error happend: no sr, y, bps... \n"
                  "[ autocorrelate_bps ] Try to use find_beat_per_second() and then repeat !")

    def plot_ac(self, ac_data, quantile, lag):
        lag = lag * 2
        #     ac_data = np.power(ac_data, 2)

        q = np.quantile(ac_data[lag:], quantile)
        m = np.mean(ac_data[lag:])

        fig, ax = plt.subplots()
        ax.plot(ac_data)
        ax.plot(np.arange(ac_data.shape[0]), [m for i in range(ac_data.shape[0])])
        ax.plot(np.arange(ac_data.shape[0]), [q for i in range(ac_data.shape[0])])
        fig.set_size_inches(30.5, 30.5)

        ax.set(title='Auto-correlation', xlabel='Lag (frames)')
        plt.grid()
        plt.show()

    def moving_average(self, observations, window=20, forecast=False):
        '''returns the smoothed version of an array of observations.'''
        cumulative_sum = np.cumsum(observations, dtype=float)
        cumulative_sum[window:] = cumulative_sum[window:] - cumulative_sum[:-window]
        if forecast:
            return np.insert(cumulative_sum[window - 1:] / window, 0, np.zeros(3))
        else:
            return cumulative_sum[window - 1:] / window

    def transform_audio_from_mp3(self, audio_without_ext, format="wav"):
        src = audio_without_ext + ".mp3"
        dst = audio_without_ext + ".wav"

        # convert wav to mp3
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")

    # Catch peaks and lows - moments where differences in music pattern appear.
    def catch_peak(self, k, tseries):
        '''

        :param k: number of low and high peaks to catch
        :param tseries: the track timeline
        :return:
        '''
        maxis = []
        minis = []
        window = tseries.shape[0] // k
        for okno in range(k):
            maximum = okno * window + tseries[okno * window: okno * window + window].argsort()[-1:][::-1]
            minimum = okno * window + tseries[okno * window: okno * window + window].argsort()[:1][::-1]
            maxis.append(maximum[0])
            minis.append(minimum[0])
            print(maximum[0], '\t', minimum[0])
        self.peaks = [maxis, minis]
        return [maxis, minis]

    def plot_peaks(self, ac_data, peaks):
        fig, ax = plt.subplots(nrows=1, sharex=True)
        # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
        #                          y_axis='log', x_axis='time', ax=ax[0])
        # ax[0].label_outer()
        ax.plot(ac_data, label='Autocorrelation')
        ax.vlines(peaks[0], ac_data.min(), ac_data.max(), label='Maxis', color='b')
        ax.vlines(peaks[1], ac_data.min(), ac_data.max(), label='Minis', color='r')
        ax.set_xticks(np.arange(ac_data.max(), step=50))
        ax.legend()
        ax.label_outer()
        fig.set_size_inches(15, 5)

    def transform_indexes_to_time(self, original_ts, transform_ts, indexes):
        len_original = original_ts.shape[0]
        len_transform = transform_ts.shape[0]
        k = len_original / len_transform
        new_peaks = [[], []]

        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                new_peaks[i].append(int(indexes[i][j] * k))

        return new_peaks

    def mel_frequency_windows(self, sorted_peaks):
        mel_array = []
        n_mfcc = 40
        n_fft = 2048
        hop_length = 512
        num_seggments = 5

        for i in range(len(sorted_peaks) - 1):
            mfcc = librosa.feature.mfcc(self.y[sorted_peaks[i]: sorted_peaks[i + 1]],
                                        sr=self.sr,
                                        n_mfcc=n_mfcc)
            mfcc = mfcc.T
            mel_array.append(mfcc)

        self.mel_windows = mel_array

    def mel_windows_avarages(self):

        mel_avarages_array = []

        for mel in self.mel_windows:
            summa = 0
            c = 0
            for i in mel:
                for j in i:
                    summa += j
                    c += 1

            average = summa / c
            mel_avarages_array.append(average)

        self.mel_avarages = mel_avarages_array

    def wave_classify(self, n_parts, plot=False):
        # parts_number = 5

        differences_between_mf = []
        quantiles_of_average_mf = []
        classes_by_mf = []  # classes of saturation for song parts, indexes of mel_avarages_array

        for i in range(1, len(self.mel_avarages)):
            differences_between_mf.append(self.mel_avarages[i] - self.mel_avarages[i - 1])

        print("MEAN ", np.mean(differences_between_mf))

        for i in range(1, n_parts + 1):
            print("Quantile {} ".format(i * 2), np.percentile(self.mel_avarages, i * 20))
            quantiles_of_average_mf.append(np.percentile(self.mel_avarages, i * 20))

        print("max mel avarages ", max(self.mel_avarages))
        print("len of mel avareges: ", len(self.mel_avarages))

        for part_index in range(len(self.mel_avarages)):
            if (self.mel_avarages[part_index] >= quantiles_of_average_mf[3]):
                print(part_index, " -- ", self.mel_avarages[part_index])
            for j in range(0, len(quantiles_of_average_mf) + 1):
                if j == 0:
                    low = min(self.mel_avarages)
                else:
                    low = quantiles_of_average_mf[j - 1]

                if (self.mel_avarages[part_index] <= quantiles_of_average_mf[j]) and (
                        self.mel_avarages[part_index] >= low):
                    classes_by_mf.append(j)
                    break

        print(classes_by_mf)

        self.classes_by_mel = classes_by_mf

        if plot:
            fig, ax = plt.subplots()
            ax.plot(differences_between_mf, label="Differences between mel.w.")
            ax.plot(self.mel_avarages, label="Mel windows avarages")
            fig.set_size_inches(15, 5)

    def transform_timest_to_autocor(self, timest_arr, as_like_arr):
        len_original = max(timest_arr)
        len_transform = len(as_like_arr)
        k = len_original / len_transform

        new_timeseries = [int(i / k) for i in timest_arr]
        return new_timeseries

    def create_dict_of_classes(self, n_parts, ac_peaks, ts, plot=True):

        dict_of_groups_time = dict()
        for c in range(n_parts):
            dict_of_groups_time[c] = []
        for j in range(len(self.classes_by_mel)):
            dict_of_groups_time[self.classes_by_mel[j]].append(ac_peaks[j])

        self.dict_of_classes = dict_of_groups_time

        if plot and n_parts == 5:

            # times = librosa.times_like(test_max)
            fig, ax = plt.subplots(nrows=1, sharex=True)
            # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
            #                          y_axis='log', x_axis='time', ax=ax[0])
            # ax[0].label_outer()
            ax.plot(ts, label='Autocorrelation')

            ax.vlines(dict_of_groups_time[0], ts.min(), ts.max(), label='0', color='#C4EC00')
            ax.vlines(dict_of_groups_time[1], ts.min(), ts.max(), label='1', color='#F5C300')
            ax.vlines(dict_of_groups_time[2], ts.min(), ts.max(), label='2', color='#FF6E6E')
            ax.vlines(dict_of_groups_time[3], ts.min(), ts.max(), label='3', color='#FF5F5F')
            ax.vlines(dict_of_groups_time[4], ts.min(), ts.max(), label='4', color='#FF0000')

            ax.set_xticks(np.arange(350, step=20))
            ax.legend()
            ax.label_outer()
            fig.set_size_inches(15, 5)
            ax.set(title='Classification of {}'.format(self.audio_file))
            fig.savefig('classified-{}.png'.format(self.audio_file))


    def plot_mel(self, window_index):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(self.mel_windows[window_index], x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set(title='MFCC window {}'.format(window_index))

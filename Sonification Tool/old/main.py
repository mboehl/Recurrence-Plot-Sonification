import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tsplayer as tsp

if __name__ == "__main__":
    with open("c:/Users/Matze/Documents/Uni/Sonification/milankovitch.txt") as file:
        data = np.transpose(np.loadtxt(file))
        time, data = data[0], data[1:]
        # normalize data
        for i in range(len(data)):
            data[i] -= np.mean(data[i])
            data[i] /= np.max(np.abs(data[i]))
        names = ["eccentricity", "climatic precession", "obliquity", "insolation quantities"]

    SAMPLE_RATE = 44100
    playback_speed = 1
    
    time_series = [tsp.TimeSeries(time, series, label, rec_cutoff=1000, rec_downsample_factor=8, rec_dim=4, rec_tau=3) for series, label in zip(data[:-1], names[:-1])]
    time_series.append(tsp.TimeSeries(time, np.sin(time), "periodic", rec_cutoff=500, rec_downsample_factor=2))

    fig = plt.figure(figsize=(16, 9), layout='constrained')
    subfigs = fig.subfigures(len(time_series) + 1, 1, height_ratios=[*(9 / len(time_series) * np.ones(len(time_series))), 1])

    time_series_players = [tsp.TSPlayer(ts, ax) for ts, ax in zip(time_series, subfigs[:len(time_series)])]


    # slider to adjust playback speed
    axes_slider_playback_speed = subfigs[-1].add_axes([.15, .5, .75, .5])
    axes_slider_playback_speed.set_xmargin(100)
    slider_playback_speed = Slider(axes_slider_playback_speed, "Playback speed", -24, 24, valinit=0, valstep=1)

    def update_pbspeed(val):
        for tsplayer in time_series_players:
            tsplayer.playback_speed = 2 ** (val / 6)
            #tsplayer.update_fm(tsplayer.slider_fm.val)

    slider_playback_speed.on_changed(update_pbspeed)

    # slider to adjust pitch shift
    axes_slider_pitch_shift = subfigs[-1].add_axes([.15, 0, .75, .5])
    axes_slider_pitch_shift.set_xmargin(100)
    slider_pitch_shift = Slider(axes_slider_pitch_shift, "Pitch shift (semitones)", -24, 24, valinit=0, valstep=1)

    def update_pitch_shift(val):
        for tsplayer in time_series_players:
            tsplayer.pitch_shift = val

    slider_pitch_shift.on_changed(update_pitch_shift)

    plt.show()
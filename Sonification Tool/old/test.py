import numpy as np
import matplotlib.pyplot as plt
from waveio import WAVWriter
from waveio.metadata import WAVMetadata
from waveio.encoding import PCMEncoding
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons
from functools import partial
import sounddevice as sd

with open("Documents/Uni/Sonification/milankovitch.txt") as file:
    data = np.transpose(np.loadtxt(file))
    time, data = data[0], data[1:]
    # normalize data
    for i in range(len(data)):
        data[i] -= np.mean(data[i])
        data[i] /= np.max(np.abs(data[i]))
    names = ["eccentricity", "climatic precession", "obliquity", "insolation quantities"]

SAMPLE_RATE = 44100
playback_speed = 1

metadata = WAVMetadata(
    frames_per_second=44100,
    num_channels=1,
    encoding=PCMEncoding(1)
)

fig, axs = plt.subplots(
        nrows=len(data),
        ncols=5,
        figsize=(16, 9),
        sharex=True,
        width_ratios=[5, 1, 1, 1, 1]
    )

# axs for time series
ts_axs = axs[:len(data)]

# plot time series
ts_plots = [
    axs[i][0].plot(time, data[i], label="Raw Signal")[0] 
    for i in range(len(data))
    ]


# frequency modulation
def freq_mod(signal, time=time, f_c=220, k=0.1):
    phi = 2 * np.pi * f_c * time + k * np.cumsum(signal)
    return .2 * np.sin(phi)

fm_data = [freq_mod(series) for series in data]
fm_plots = [
    axs[i][0].plot(time, fm_data[i], linewidth=.8, zorder=0, label="FM Signal")[0] 
    for i in range(len(data))
    ]

plot_colors = [ts_plots[0].get_color(), fm_plots[0].get_color()]
plot_labels = ["Raw Signal", "FM Signal"]


# check boxes to show different plots
axs_checkbox_visibility = [ax[0].inset_axes([0.0, 0.0, 0.22, 0.2]) for ax in ts_axs]
checkboxes = [
    CheckButtons(
        ax=ax,
        labels=plot_labels,
        actives=[plot.get_visible() for plot in plots],
        label_props={'color': plot_colors},
        frame_props={'edgecolor': plot_colors},
        check_props={'facecolor': plot_colors},
    )
    for plots, ax in zip(zip(ts_plots, fm_plots), axs_checkbox_visibility)
]

def update_plot_visibility(index, label):
    plot = [ts_plots[index], fm_plots[index]][plot_labels.index(label)]
    plot.set_visible(not plot.get_visible())
    plot.figure.canvas.draw_idle()

for i in range(len(data)):
    axs[i][0].set_title(names[i])   
    checkboxes[i].on_clicked(partial(update_plot_visibility, i))





# slider for playback speed
ax_slider_pbspeed = fig.add_axes((.2, 0, .65, .05))
slider_pbspeed = Slider(ax_slider_pbspeed, "Playback Speed", valmin=0, valmax=2, valinit=1)

def update_pbspeed(val):
    global playback_speed
    playback_speed = val

slider_pbspeed.on_changed(update_pbspeed)


# slider for fm sensitivity
ax_slider_fmsens = fig.add_axes((.2, .05, .65, .05))
slider_fmsens = Slider(ax_slider_fmsens, "FM Sensitivity", valmin=0.001, valmax=1, valinit=0.1)

def update_fmsens(val):
    global fm_data
    fm_data = [freq_mod(series, k=val) for series in data]
    for plot, fm_signal in zip(fm_plots, fm_data):
        plot.set(ydata=fm_signal)
        plot.figure.canvas.draw_idle()

slider_fmsens.on_changed(update_fmsens)


# buttons to save as wav
def save_wav(index, event):
    with WAVWriter(metadata, f"Documents/Uni/Sonification/milankovitch_sound_{names[index]}.wav") as wav:
        wav.append_channels(data[index])

save_buttons = [Button(ax[-1], "Save .wav") for ax in axs[:len(data)]]

for i in range(len(save_buttons)):
    save_buttons[i].on_clicked(partial(save_wav, i))


# buttons for playback
def play_signal(signal, event):
    sd.play(signal, SAMPLE_RATE * playback_speed)

play_buttons = [Button(ax[1], "Raw Signal") for ax in axs]

for i in range(len(play_buttons)):
    play_buttons[i].on_clicked(partial(play_signal, data[i]))


# buttons for freq mod playback
play_fm_buttons = [Button(ax[2], "FM Signal") for ax in axs]

def play_fm_signal(index, event):
    play_signal(fm_data[index], event)

for i in range(len(play_fm_buttons)):
    play_fm_buttons[i].on_clicked(partial(play_fm_signal, i))


# buttons stop playback
stop_buttons = [Button(ax[-2], "Stop Playback") for ax in axs]

for button in stop_buttons:
    button.on_clicked(lambda x: sd.stop())

plt.show()
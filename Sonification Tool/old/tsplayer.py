import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from timeseries import TimeSeries
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import TextBox
import sounddevice as sd
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import librosa
import utils

class TSPlayer():
    SAMPLE_RATE = 44100
    playback_speed = 1
    pitch_shift = 0
    ts : TimeSeries

    def __init__(self, ts: TimeSeries, fig: matplotlib.figure.FigureBase):
        self.ts = ts
        self.fig = fig
        self.current_scene = "graph"
        self.set_scene("recurrence plot")

        plt.connect("button_press_event", self.on_click)
        

    def set_scene(self, target: str):
        self.fig.clear
        self.current_scene = target

        if target == "graph":
            self.axes = self.fig.subplots(1, 6, width_ratios=[3, 1, 1, 1, 1, 10])
            
            # information text
            self.info = self.axes[0]
            self.info.set_axis_off()
            self.info.text(0, 1, self.ts.label, horizontalalignment='left', verticalalignment='top', wrap=True)

            # plotting axes
            self.plot_axes = self.axes[-1]
            # plot raw ts
            self.plot = self.plot_axes.plot(self.ts.time, self.ts.data, label=self.ts.label + " (raw)")[0]

            # plot freq modulated ts
            self.fm_plot = self.plot_axes.plot(self.ts.time, self.ts.fm_data, label=self.ts.label + " (fm)", zorder=0)[0]
            self.fm_plot.set_visible(False)

            # check boxes to show different plots
            plot_colors = [self.plot.get_color(), self.fm_plot.get_color()]
            self.plot_labels = ["Raw Signal", "FM Signal"]

            self.axs_checkbox_visibility = self.plot_axes.inset_axes([0.0, 0.0, 0.22, 0.2])
            self.checkboxes = CheckButtons(
                    ax=self.axs_checkbox_visibility,
                    labels=self.plot_labels,
                    actives=[plot.get_visible() for plot in [self.plot, self.fm_plot]],
                    label_props={'color': plot_colors},
                    frame_props={'edgecolor': plot_colors},
                    check_props={'facecolor': plot_colors},
                )

            self.checkboxes.on_clicked(self.update_plot_visibility)

            # playback buttons
            self.button_stop = Button(self.axes[1], "\u25A0")
            self.button_stop.on_clicked(self.play_stop)

            self.button_play_raw = Button(self.axes[2], "\u25B6 raw")
            self.button_play_raw.on_clicked(self.play_raw)

            self.button_play_fm = Button(self.axes[3], "\u25B6 fm")
            self.button_play_fm.on_clicked(self.play_fm)

            self.slider_fm = Slider(self.axes[4], "FM sensitivity", valmin=0, valmax=.1, valinit=.04, orientation='vertical')
            self.slider_fm.on_changed(self.update_fm)

        elif target == "recurrence plot":
            self.axes = self.fig.subplots(1, 5, width_ratios=[3, 2, 2, 2, 2])

            # information text
            self.info = self.axes[0]
            self.info.set_axis_off()
            self.info.text(0, 1, self.ts.label, horizontalalignment='left', verticalalignment='top', wrap=True)

            # recurrence plot parameters
            # self.rec_params = TextBox(self.axes[3], "Recurrence plot parameters\n (downsample_factor, \ncutoff, \ndim, \ntau, \nrecurrence_rate)", initial=f"({self.ts.rec_downsample_factor}, {self.ts.rec_cutoff}, {self.ts.rec_data.dim}, {self.ts.rec_data.tau}, {self.ts.rec_data.recurrence_rate():.3f})")

            # def update_rec_params(text):
            #     try:
            #         downsample_factor, cutoff, dim, tau, recurrence_rate = eval(text)
            #         self.ts.rec_data = RecurrencePlot((self.ts.data[::downsample_factor])[:cutoff], dim=dim, tau=tau, recurrence_rate=recurrence_rate)
            #         self.rec_plot.set_data(self.ts.recurrence_matrix())
            #         self.rec_plot.figure.canvas.draw_idle()
            #     except Exception as e:
            #         print(f"Error updating recurrence plot parameters: {e}")

            # self.button_update_rec = Button(self.axes[2], "\u25B6")
            # self.button_update_rec.on_clicked(lambda event: update_rec_params(self.rec_params.text))

            # recurrence plot
            self.plot_axes = self.axes[-3]
            self.rec_plot = self.plot_axes.imshow(self.ts.recurrence_matrix(), cmap="binary", origin="lower")

            # read upper half of recurrence plot as spectrogram and sonify
            self.rec_halfed_rotated = utils.upper_triangle_aligned(self.ts.recurrence_matrix())          
            self.axes[-2].imshow(self.rec_halfed_rotated, origin="lower")
            self.axes[-1].imshow(np.fliplr(self.rec_halfed_rotated.T), origin="lower")

            # playback buttons
            self.button_stop = Button(self.axes[1], "\u25A0")
            self.button_stop.on_clicked(self.play_stop)


    def update_plot_visibility(self, label: str):
            plot = [self.plot, self.fm_plot][self.plot_labels.index(label)]
            plot.set_visible(not plot.get_visible())
            plot.figure.canvas.draw_idle()

    def play_signal(self, signal: np.array, time_scale=1):
        if self.pitch_shift != 0:
            signal = librosa.effects.pitch_shift(signal, n_steps=self.pitch_shift, sr=self.SAMPLE_RATE, res_type="soxr_vhq")
        sd.play(.8 * utils.normalize(signal), self.SAMPLE_RATE * self.playback_speed * time_scale)
    
    def play_raw(self, event):
        self.play_signal(self.ts.data)

    def play_fm(self, event):
        self.play_signal(self.ts.fm_data, time_scale=.5)

    def play_rec(self, event):
        self.rec_halfed_rotated = utils.upper_triangle_aligned(self.ts.recurrence_matrix())
        self.play_signal(self.spectrogram_to_audio(self.rec_halfed_rotated))

    def play_rec_diagonals_as_horizontal_lines(self, event):
        spec = np.fliplr(utils.upper_triangle_aligned(self.ts.recurrence_matrix()).T)
        self.play_signal(self.spectrogram_to_audio(spec, pad_top_freq_factor=1.7))

    def play_rec_diaonals_as_vertical_lines(self, event):
        spec = utils.upper_triangle_aligned(self.ts.recurrence_matrix())
        self.play_signal(self.spectrogram_to_audio(spec, pad_top_freq_factor=1.7))

    def play_stop(self, event):
        sd.stop()

    def spectrogram_to_audio(self, spec, pad_top_freq_factor=0):
        spec = np.pad(spec, ((0, int(pad_top_freq_factor * len(spec))), (0, 0)), mode='constant', constant_values=0)
        n = len(spec) - 1
        fft = ShortTimeFFT(win=hann(2 * n, sym=False), hop=n//2, fs=41000, fft_mode="onesided")
        return fft.istft(spec[:fft.f_pts])

    def generate_tone(self, frequency, duration):
        t = np.arange(duration * self.SAMPLE_RATE)
        return np.sin(2 * np.pi * frequency * t)

    def update_fm(self, val:float):
        self.ts.fm_data = self.ts.frequency_modulation(freq=440 / self.playback_speed, sens=val)
        self.fm_plot.set(ydata=self.ts.fm_data)
        self.fm_plot.figure.canvas.draw_idle()
    
    def on_click(self, event):
        if self.current_scene == "recurrence plot":
            if event.inaxes == self.plot_axes:
                self.fig_enlarge, self.ax_enlarge = plt.subplots(1, 3, layout="constrained")
                self.ax_enlarge[0].imshow(self.ts.recurrence_matrix(), cmap="binary", origin="lower")
                self.ax_enlarge[1].imshow(utils.half_diagonals_as_columns(self.ts.recurrence_matrix()), origin="lower")
                self.ax_enlarge[2].imshow(self.ts.recurrence_extract_diagonal_lines().T, origin="lower")
                plt.show()

            if event.inaxes == self.axes[-2]:
                self.play_rec_diaonals_as_vertical_lines(event)

            if event.inaxes == self.axes[-1]:
                self.play_rec_diagonals_as_horizontal_lines(event)


    
if __name__ == "__main__":

    def normalize(signal):
        return signal / np.max(np.abs(signal))
    
    fig2 = plt.figure(figsize=(3, 2), layout='constrained')
    time = np.arange(5000)
    ts = TimeSeries(time, np.sin(0.1 * time), "periodic", rec_cutoff=1000, rec_downsample_factor=3)
    tsp = TSPlayer(ts, fig2)
    plt.connect("button_press_event", tsp.on_click)

    spec = utils.upper_triangle_aligned(ts.recurrence_matrix())
    k = 3
    spec = utils.squish_matrix(np.pad(np.fliplr(spec), ((0, int(k * len(spec))), (0, 0)), mode='constant', constant_values=0), factor=1)
    n = len(spec) - 1
    fft = ShortTimeFFT(win=hann(2 * n, sym=False), hop=int(n/2), fs=1000, fft_mode="onesided")
    audio = normalize(fft.istft(spec[:fft.f_pts]))
    
    fig = plt.figure(figsize=(5, 3))
    axs = fig.subplots(1, 3)
    axs[0].imshow(ts.recurrence_matrix(), cmap="binary", origin="lower")
    axs[1].imshow(spec, origin="lower")
    axs[2].imshow(abs(fft.stft(audio)), origin="lower")

    def play(event):
        tsp.play_signal(audio)

    plt.connect("button_press_event", play)
    plt.show()


    
    pass
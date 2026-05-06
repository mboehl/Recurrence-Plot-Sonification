import numpy as np
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

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton, QToolButton

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

class TSPlayer(QtWidgets.QWidget):
    SAMPLE_RATE : int = 44100
    playback_speed : float = 1.0
    pitch_shift : int = -12
    ts : TimeSeries

    def __init__(self, ts: TimeSeries):
        super().__init__()
        self.ts = ts    

        # widgets    
        self.label = QLabel()
        self.title = QLabel(self.ts.label)
        self.title.setMaximumWidth(200)

        self.fig = Figure(figsize=(5, 3), layout='constrained')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(200)
        
        
        # selection tool bar
        self.toolbar_select = QHBoxLayout()
        self.button_select_full = QPushButton("Full plot")
        self.button_select_square = QRadioButton("□")
        self.button_select_square.setStyleSheet("font-size: 20px;")
        self.button_select_square.setChecked(True)
        self.button_select_tilted_square = QRadioButton("◇ (lossy)")
        self.button_select_parallelogram = QRadioButton("▱")
        self.button_select_parallelogram.setStyleSheet("font-size: 30px;")
        self.toolbar_select.addWidget(self.button_select_full)
        self.toolbar_select.addWidget(self.button_select_square)
        self.toolbar_select.addWidget(self.button_select_tilted_square)
        self.toolbar_select.addWidget(self.button_select_parallelogram)
        
        self.selection_tool = "square"
        self.button_select_full.clicked.connect(self.select_full_plot)
        self.button_select_square.clicked.connect(lambda: setattr(self, "selection_tool", "square"))
        self.button_select_tilted_square.clicked.connect(lambda: setattr(self, "selection_tool", "tilted_square"))
        self.button_select_parallelogram.clicked.connect(lambda: setattr(self, "selection_tool", "parallelogram"))


        self.selected_area = None
        self.sonification_data = None
        # later initialized with full plot selected

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        # matplotlib toolbar for the plot
        self.toolbar_plt = NavigationToolbar(self.canvas, self)
        self.toolbar_plt.setMaximumHeight(30)

        # top level layout
        self.plot_layout = QVBoxLayout()
        self.plot_layout.addLayout(self.toolbar_select)
        self.plot_layout.addWidget(self.canvas)
        self.plot_layout.addWidget(self.toolbar_plt)

        self.h_layout = QHBoxLayout(self)
        self.h_layout.addWidget(self.label)
        self.h_layout.addWidget(self.title)
        self.h_layout.addLayout(self.plot_layout)
        # self.h_layout.addWidget(self.pixmap)


        self.current_scene = "linear"
        self.set_scene("recurrence plot")
    

    def set_scene(self, target: str):
        self.fig.clear
        self.current_scene = target

        if target == "linear":
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


            self.plot_axes = self.fig.subplots()

            # recurrence plot
            self.rec_plot = self.plot_axes.imshow(self.ts.recurrence_matrix(), cmap="binary", origin="lower")

            self.fig_sonification = Figure(figsize=(5, 3), layout='constrained')
            self.canvas_sonification = FigureCanvas(self.fig_sonification)
            self.axes_sonification = self.fig_sonification.subplots(1, 2)
            self.canvas_sonification.mpl_connect("button_press_event", self.on_press)
            
            # read upper half of recurrence plot as spectrogram and sonify
            self.sonification_data = utils.matrix_extract_square(self.ts.recurrence_matrix(), self.square((0, 0), self.ts.recurrence_matrix().shape))
            self.update_plots_sonification()

            self.h_layout.addWidget(self.canvas_sonification)


            # self.pixmap.setPixmap(QPixmap(utils.matrix_to_qimage(self.ts.recurrence_matrix())))

            

            
            # playback buttons
            self.button_stop = QPushButton("\u25A0")
            self.button_stop.clicked.connect(self.play_stop)
            self.button_stop.setMaximumWidth(50)
            self.h_layout.addWidget(self.button_stop)


    def update_plot_visibility(self, label: str):
            plot = [self.plot, self.fm_plot][self.plot_labels.index(label)]
            plot.set_visible(not plot.get_visible())
            plot.figure.canvas.draw_idle()

    def play_signal(self, signal: np.array, time_scale=1):
        if self.pitch_shift != 0:
            signal = librosa.effects.pitch_shift(signal, n_steps=self.pitch_shift, sr=self.SAMPLE_RATE, res_type="soxr_vhq")
        sd.play(.5 * utils.normalize(signal), self.SAMPLE_RATE * self.playback_speed * time_scale)
    
    def play_raw(self, event):
        self.play_signal(self.ts.data)

    def play_fm(self, event):
        self.play_signal(self.ts.fm_data, time_scale=.5)

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

    def parallelogram(self, start, end):
        height = end[1] - start[1]
        width = end[0] - start[0] - height
        return np.array([start, (start[0] + width, start[1]), end, (start[0] + height, start[1] + height)])
    
    def square(self, start, end):
        width = end[0] - start[0]
        height = end[1] - start[1]
        side_length = min(abs(width), abs(height))
        width = side_length * np.sign(width)
        height = side_length * np.sign(height)
        return np.array([start, (start[0] + width, start[1]), (start[0] + width, start[1] + height), (start[0], start[1] + height)])
    
    def tilted_square(self, start, end):
        width = end[0] - start[0]
        if width % 2 != 0:
            width += 1
        return np.array([start, (start[0] + width//2, start[1] + width//2), (start[0] + width, start[1]), (start[0] + width//2, start[1] - width//2)])

    
    def on_press(self, event):
        if self.current_scene == "recurrence plot":
            if event.inaxes == self.plot_axes:
                if self.selected_area:
                    self.selected_area.remove()
                self.selected_area_start = (int(event.xdata), int(event.ydata))
                match self.selection_tool:
                    case "square":
                        self.selected_area = self.plot_axes.add_patch(Polygon(self.square(self.selected_area_start, self.selected_area_start), closed=True, fill=False, edgecolor='cyan', linewidth=2))
                    case "tilted_square":
                        self.selected_area = self.plot_axes.add_patch(Polygon(self.tilted_square(self.selected_area_start, self.selected_area_start), closed=True, fill=False, edgecolor='cyan', linewidth=2))
                    case "parallelogram":
                        self.selected_area = self.plot_axes.add_patch(Polygon(self.parallelogram(self.selected_area_start, self.selected_area_start), closed=True, fill=False, edgecolor='cyan', linewidth=2))

            elif event.inaxes == self.axes_sonification[0]:
                self.play_signal(self.spectrogram_to_audio(self.sonification_data, pad_top_freq_factor=3))

            elif event.inaxes == self.axes_sonification[1]:
                self.play_signal(self.spectrogram_to_audio(np.fliplr(self.sonification_data).T, pad_top_freq_factor=3))

    def on_motion(self, event):
        if self.current_scene == "recurrence plot" and event.buttons and self.selected_area and event.inaxes == self.plot_axes:
            current_pos = (int(event.xdata), int(event.ydata))
            match self.selection_tool:
                case "square":
                    self.selected_area.set_xy(self.square(self.selected_area_start, current_pos))
                case "tilted_square":
                    self.selected_area.set_xy(self.tilted_square(self.selected_area_start, current_pos))
                case "parallelogram":
                    self.selected_area.set_xy(self.parallelogram(self.selected_area_start, current_pos))

            self.selected_area.figure.canvas.draw_idle()

    def on_release(self, event):
        if self.current_scene == "recurrence plot" and self.selected_area:
            selected_area_end = (int(event.xdata), int(event.ydata))
            match self.selection_tool:
                case "square":
                    self.sonification_data = utils.matrix_extract_square(self.ts.recurrence_matrix(), self.square(self.selected_area_start, selected_area_end))
                case "tilted_square":
                    self.sonification_data = utils.matrix_extract_tilted_square(self.ts.recurrence_matrix(), self.tilted_square(self.selected_area_start, selected_area_end))
                case "parallelogram":
                    self.sonification_data = utils.matrix_extract_parallelogram(self.ts.recurrence_matrix(), self.parallelogram(self.selected_area_start, selected_area_end))
            self.update_plots_sonification()
        self.plot_axes.figure.canvas.draw_idle()

    def select_full_plot(self, event):
        self.sonification_data = self.ts.recurrence_matrix()
        self.update_plots_sonification()
        self.selection_tool = "square"
        self.button_select_square.setChecked(True)


    def update_plots_sonification(self):
        self.axes_sonification[0].imshow(self.sonification_data, origin="lower")
        self.axes_sonification[1].imshow(np.fliplr(self.sonification_data).T, origin="lower")
        self.fig_sonification.canvas.draw_idle()

    
if __name__ == "__main__":

    def normalize(signal):
        return signal / np.max(np.abs(signal))
    
    fig2 = plt.figure(figsize=(3, 2), layout='constrained')
    time = np.arange(5000)
    ts = TimeSeries(time, np.sin(0.1 * time), "periodic", rec_cutoff=1000, rec_downsample_factor=3)
    tsp = TSPlayer(ts, fig2)
    plt.connect("button_press_event", tsp.on_press)

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
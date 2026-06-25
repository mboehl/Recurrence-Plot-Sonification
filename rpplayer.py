import numpy as np
import matplotlib.pyplot as plt
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
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton, QToolButton, QCheckBox, QButtonGroup, QToolBar, QSlider
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

class RPPlayer(QtWidgets.QWidget):
    sample_rate : int = 44100
    #freq_min : int = 20
    max_freq : int = 1000
    playback_speed : float = 1.0
    pitch_shift : int = 0
    volume : float = 0.8
    label : str

    distance_matrix : np.array
    recurrence_matrix : np.array
    sonification_matrix : np.array # matrix used for sonification, either recurrence or distance matrix depending on user selection
    sonification_data : np.array # extracted part of the sonification matrix that is actually sonified
    use_recurrence_plot : bool = True

    def __init__(self, matrix : np.array, label : str, is_rec_matrix: bool = True):
        super().__init__()
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Input matrix must be square, but got shape {matrix.shape}.")
        
        if is_rec_matrix:
            self.recurrence_matrix = matrix.astype(int)
            self.distance_matrix = None
        else:
            self.distance_matrix = matrix
            threshold = np.percentile(self.distance_matrix, 5)
            self.recurrence_matrix = (self.distance_matrix < threshold).astype(int)
            # apply function 1 / (1 + x) to distance matrix to get values in range (0, 1] where 1 corresponds to distance 0 and values close to 0 correspond to large distances
            self.rescaled_distance_matrix = 1 / (1 + 5 * self.distance_matrix)
            
        self.sonification_matrix = self.recurrence_matrix

        

        self.label = label.capitalize()

        # widgets
        self.title = QLabel(self.label)
        self.title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")

        self.fig = Figure(figsize=(3, 3), layout='constrained')
        self.fig.set_facecolor("#f0f0f0")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(200)
        
        # slider for recurrence threshold if distance matrix is available
        if self.distance_matrix is not None:
            self.slider_threshold = QSlider(Qt.Horizontal)
            self.slider_threshold.setRange(0, 50)
            self.slider_threshold.setValue(5)
            self.slider_threshold.setTickInterval(5)
            self.slider_threshold.setTickPosition(QSlider.TicksBelow)
            self.slider_threshold.valueChanged.connect(self.on_threshold_changed)
            self.label_threshold = QLabel(f"Recurrence threshold: {self.slider_threshold.value()}-th percentile of distances")
        
        # selection tool bar
        self.toolbar_select_predefined = QToolBar()
        self.toolbar_select = QToolBar() #QHBoxLayout()
        self.button_select_full = QPushButton("Full plot")
        self.button_select_full_triangle_rotated = QPushButton("Full triangle rotated")
        self.button_select_full_triangle_sheared = QPushButton("Full triangle sheared")
        self.button_select_triangle_rotated = QRadioButton("△")
        self.button_select_triangle_sheared = QRadioButton("◸")
        self.button_select_square = QRadioButton("□")
        self.button_select_square.setStyleSheet("font-size: 20px;")
        self.button_select_square.setChecked(True)
        self.button_select_tilted_square = QRadioButton("◇")
        self.button_select_parallelogram = QRadioButton("▱")
        self.button_select_parallelogram.setStyleSheet("font-size: 30px;")
        
        self.toolbar_select_predefined.addWidget(self.button_select_full)
        self.toolbar_select_predefined.addWidget(self.button_select_full_triangle_rotated)
        self.toolbar_select_predefined.addWidget(self.button_select_full_triangle_sheared)
        self.toolbar_select.addWidget(self.button_select_square)
        self.toolbar_select.addWidget(self.button_select_tilted_square)
        self.toolbar_select.addWidget(self.button_select_triangle_rotated)
        self.toolbar_select.addWidget(self.button_select_parallelogram)
        self.toolbar_select.addWidget(self.button_select_triangle_sheared)
        
        self.selection_tool = "square"
        self.button_select_full.clicked.connect(self.select_full_plot)
        self.button_select_full_triangle_rotated.clicked.connect(self.select_full_triangle_rotated)
        self.button_select_full_triangle_sheared.clicked.connect(self.select_full_triangle_sheared)
        self.button_select_triangle_rotated.clicked.connect(lambda: setattr(self, "selection_tool", "triangle_rotated"))
        self.button_select_triangle_sheared.clicked.connect(lambda: setattr(self, "selection_tool", "triangle_sheared"))
        self.button_select_square.clicked.connect(lambda: setattr(self, "selection_tool", "square"))
        self.button_select_tilted_square.clicked.connect(lambda: setattr(self, "selection_tool", "tilted_square"))
        self.button_select_parallelogram.clicked.connect(lambda: setattr(self, "selection_tool", "parallelogram"))

        self.selected_area = None
        # initially select full plot
        self.sonification_data = self.sonification_matrix
        
        # connect mouse events for selecting area to sonify
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

        
        # matplotlib toolbar for the plot
        # self.toolbar_plt = NavigationToolbar(self.canvas, self)
        # self.toolbar_plt.setMaximumHeight(30)

        
        
        # playback controls
        # slider for adjusting volume
        self.slider_volume = QSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(80)
        self.label_volume = QLabel(f"Volume: {self.slider_volume.value()}%")
        self.slider_volume.valueChanged.connect(self.on_volume_changed)

        # slider for adjusting playback speed
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 100)
        self.slider_speed.setValue(10)
        self.label_speed = QLabel(f"Playback speed: {self.slider_speed.value() / 10:.1f}x")
        self.slider_speed.valueChanged.connect(self.on_speed_changed)

        # slider for adjusting maximum frequency of sonification
        self.slider_max_freq = QSlider(Qt.Horizontal)
        self.slider_max_freq.setRange(500, 2000)
        self.slider_max_freq.setValue(1000)
        self.label_max_freq = QLabel(f"Max frequency: {self.slider_max_freq.value()} Hz")
        self.slider_max_freq.valueChanged.connect(self.on_max_freq_changed)

        # layout for playback controls
        self.playback_controls_layout = QVBoxLayout()
        self.playback_controls_layout.addWidget(self.label_volume)
        self.playback_controls_layout.addWidget(self.slider_volume)
        self.playback_controls_layout.addWidget(self.label_speed)
        self.playback_controls_layout.addWidget(self.slider_speed)
        self.playback_controls_layout.addWidget(self.label_max_freq)
        self.playback_controls_layout.addWidget(self.slider_max_freq)


        
        # plots
        # sonification matrix plot
        self.plot_axes = self.fig.subplots()        
        
        self.rec_plot = self.plot_axes.imshow(self.sonification_matrix, cmap="binary", origin="lower")
        self.plot_axes.set_xticks([])
        self.plot_axes.set_yticks([])
        self.plot_axes.set_xlabel("Time")
        self.plot_axes.set_ylabel("Time")

        # sonification plot / spectrograms
        self.fig_sonification = Figure(figsize=(6, 3), layout='constrained')
        self.fig_sonification.set_facecolor("#f0f0f0")
        self.canvas_sonification = FigureCanvas(self.fig_sonification)

        # create the two subplots for the sonification data and its rotated version
        self.axes_sonification = self.fig_sonification.subplots(1, 2)
        self.sonification_img_1 = self.axes_sonification[0].imshow(self.sonification_data, origin="lower")
        self.sonification_img_2 = self.axes_sonification[1].imshow(np.rot90(self.sonification_data), origin="lower")

        # set labels and ticks for the sonification plots
        self.axes_sonification[0].set_xticks([])
        self.axes_sonification[0].set_yticks([])
        self.axes_sonification[0].set_xlabel("Time")
        self.axes_sonification[0].set_ylabel("Frequency")
        self.axes_sonification[1].set_xticks([])
        self.axes_sonification[1].set_yticks([])
        self.axes_sonification[1].set_xlabel("Time")
        self.axes_sonification[1].set_ylabel("Frequency")

        # connect mouse events for playing the sonification data when clicking on the plots
        self.canvas_sonification.mpl_connect("button_press_event", self.on_press)
        


        
        # switch to use recurrence plot or distance matrix for sonification
        if self.distance_matrix is not None:
            self.layout_matrix_selection = QHBoxLayout()
            self.check_use_recurrence_plot = QRadioButton("Use recurrence plot", self)
            self.check_use_recurrence_plot.setChecked(True)
            self.check_use_distance_matrix = QRadioButton("Use distance matrix", self)
            self.check_use_distance_matrix.setChecked(False)
            self.layout_matrix_selection.addWidget(self.check_use_recurrence_plot)
            self.layout_matrix_selection.addWidget(self.check_use_distance_matrix)


        # layout for recurrence plot and controls
        self.plot_layout = QVBoxLayout()
        self.plot_layout.addWidget(self.title)
        if self.distance_matrix is not None:
             self.plot_layout.addLayout(self.layout_matrix_selection)
             self.check_use_recurrence_plot.toggled.connect(self.on_use_recurrence_plot_changed)
        self.plot_layout.addWidget(self.toolbar_select_predefined)
        self.plot_layout.addWidget(self.toolbar_select)
        self.plot_layout.addWidget(self.canvas)
        self.plot_layout.addWidget(self.label_threshold)
        self.plot_layout.addWidget(self.slider_threshold)
        #self.plot_layout.addWidget(self.toolbar_plt)
        
        
        # button to stop playback
        self.button_stop = QPushButton("\u25A0")
        self.button_stop.clicked.connect(self.play_stop)
        self.button_stop.setMaximumWidth(50)



        # top level horizontal layout
        self.h_layout = QHBoxLayout(self)
        self.h_layout.addLayout(self.plot_layout)
        self.h_layout.addSpacing(20)
        self.h_layout.addLayout(self.playback_controls_layout)
        self.h_layout.addSpacing(20)
        self.h_layout.addWidget(self.button_stop)
        self.h_layout.addSpacing(20)
        self.h_layout.addWidget(self.canvas_sonification)


        


    def play_signal(self, signal: np.array):
        if self.pitch_shift != 0:
            signal = librosa.effects.pitch_shift(signal, n_steps=self.pitch_shift, sr=self.sample_rate, res_type="soxr_vhq")
        sd.play(self.volume * utils.normalize(signal), self.sample_rate * self.playback_speed)

    def play_stop(self, event):
        sd.stop()

    def spectrogram_to_audio(self, spec, max_freq=None):
        if max_freq is None:
            max_freq = self.max_freq
        n = len(spec) - 1
        fft = ShortTimeFFT(win=hann(2 * n, sym=False),
                            hop=n // 2,
                            fs=2 * max_freq,
                            fft_mode="onesided")
        audio = fft.istft(spec[:fft.f_pts])


        return audio #librosa.resample(audio, orig_sr=max_freq, target_sr=self.sample_rate)

    def _parallelogram(self, start, end):
        height = end[1] - start[1]
        width = end[0] - start[0] - height
        return np.array([start, (start[0] + width, start[1]), end, (start[0] + height, start[1] + height)])
    
    def _square(self, start, end):
        width = end[0] - start[0]
        height = end[1] - start[1]
        side_length = min(abs(width), abs(height))
        width = side_length * np.sign(width)
        height = side_length * np.sign(height)
        return np.array([start, (start[0] + width, start[1]), (start[0] + width, start[1] + height), (start[0], start[1] + height)])
    
    def _tilted_square(self, start, end):
        width = end[0] - start[0]
        if width % 2 != 0:
            width += 1
        return np.array([start, (start[0] + width//2, start[1] + width//2), (start[0] + width, start[1]), (start[0] + width//2, start[1] - width//2)])
    
    def _triangle(self, start, end):
        width = end[0] - start[0]
        height = end[1] - start[1]
        cathetus_length = max(abs(width), abs(height))
        if width > 0:
            return np.array([start, (start[0], start[1] + cathetus_length), (start[0] + cathetus_length, start[1] + cathetus_length)])
        else:
            return np.array([start, (start[0] - cathetus_length, start[1]), (start[0] - cathetus_length, start[1] - cathetus_length)])

    
    def on_press(self, event):
        # check if the click is inside the recurrence plot axes or the sonification axes
        if event.inaxes == self.plot_axes:
            # start drawing the selection area
            if self.selected_area:
                self.selected_area.remove()
            self.selected_area_start = (int(event.xdata), int(event.ydata))
            self.selected_area = self.plot_axes.add_patch(Polygon(np.array([self.selected_area_start]*4), closed=True, fill=False, edgecolor='cyan', linewidth=2))
        
        elif event.inaxes == self.axes_sonification[0]:
            # play the sonification data corresponding to the selected area
            self.play_signal(self.spectrogram_to_audio(self.sonification_data))

        elif event.inaxes == self.axes_sonification[1]:
            self.play_signal(self.spectrogram_to_audio(np.rot90(self.sonification_data)))

    def on_motion(self, event):
        if event.buttons and self.selected_area and event.inaxes == self.plot_axes:
            current_pos = (int(event.xdata), int(event.ydata))
            match self.selection_tool:
                case "square":
                    self.selected_area.set_xy(self._square(self.selected_area_start, current_pos))
                case "tilted_square":
                    self.selected_area.set_xy(self._tilted_square(self.selected_area_start, current_pos))
                case "parallelogram":
                    self.selected_area.set_xy(self._parallelogram(self.selected_area_start, current_pos))
                case "triangle_rotated":
                    self.selected_area.set_xy(self._triangle(self.selected_area_start, current_pos))
                case "triangle_sheared":
                    self.selected_area.set_xy(self._triangle(self.selected_area_start, current_pos))

            self.selected_area.figure.canvas.draw_idle()

    def on_release(self, event):
        if self.selected_area:
            self.selected_area_end = (int(event.xdata), int(event.ydata))
            self.update_sonification_data()
            self.update_plots_sonification()
        self.plot_axes.figure.canvas.draw_idle()

    def _calculate_threshold(self):
        if self.distance_matrix is not None:
            return np.percentile(self.distance_matrix, self.slider_threshold.value())
        else:
            return None

    def on_threshold_changed(self, event):
        self.label_threshold.setText(f"Recurrence threshold: {self.slider_threshold.value()}-th percentile of distances")
        threshold = self._calculate_threshold()
        self.recurrence_matrix = (self.distance_matrix < threshold).astype(int)
        if self.use_recurrence_plot:
            self.sonification_matrix = self.recurrence_matrix
        else:
            # use the rescaled distance matrix for sonification, but only include values where the distance is below the threshold (i.e., where there is a recurrence according to the current threshold)
            self.sonification_matrix = self.rescaled_distance_matrix * (self.distance_matrix < threshold)
        
        self.update_sonification_data()
        self.update_plots_sonification()

    def on_volume_changed(self, event):
        self.label_volume.setText(f"Volume: {self.slider_volume.value()}%")
        self.volume = self.slider_volume.value() / 100

    def on_speed_changed(self, event):
        self.label_speed.setText(f"Playback speed: {self.slider_speed.value() / 10:.1f}x")
        self.playback_speed = self.slider_speed.value() / 10
        self.pitch_shift = - np.round(12 * np.log2(self.playback_speed)).astype(int)

    def on_max_freq_changed(self, event):
        self.label_max_freq.setText(f"Max frequency: {self.slider_max_freq.value()} Hz")
        self.max_freq = self.slider_max_freq.value()
        self.sample_rate = self.max_freq * 2

    def on_use_recurrence_plot_changed(self, event):
        self.use_recurrence_plot = self.check_use_recurrence_plot.isChecked()
        if self.use_recurrence_plot:
            self.sonification_matrix = self.recurrence_matrix
            self.rec_plot.set_cmap("binary")
        else:
            self.sonification_matrix = self.rescaled_distance_matrix
            self.rec_plot.set_cmap("magma")
        self.update_sonification_data()
        self.update_plots_sonification()

    def update_sonification_data(self):
        if self.selected_area is None:
            self.sonification_data = self.sonification_matrix
        else:
            if self.selection_tool == "square":
                self.sonification_data = utils.matrix_extract_square(self.sonification_matrix, self._square(self.selected_area_start, self.selected_area_end))
            elif self.selection_tool == "tilted_square":
                self.sonification_data = utils.matrix_extract_tilted_square(self.sonification_matrix, self._tilted_square(self.selected_area_start, self.selected_area_end))
            elif self.selection_tool == "parallelogram":
                self.sonification_data = utils.matrix_extract_parallelogram(self.sonification_matrix, self._parallelogram(self.selected_area_start, self.selected_area_end))
            elif self.selection_tool == "triangle_rotated":
                self.sonification_data = utils.half_diagonals_as_columns(utils.matrix_extract_square(self.sonification_matrix, self._square(self.selected_area_start, self.selected_area_end)).T, dtype=np.float32)
            elif self.selection_tool == "triangle_sheared":
                self.sonification_data = utils.upper_triangle_aligned(utils.matrix_extract_square(self.sonification_matrix, self._square(self.selected_area_start, self.selected_area_end)).T)

    def select_full_plot(self, event):
        if self.selected_area is not None:
            self.selected_area.remove()
            self.selected_area = None
        self.update_sonification_data()
        self.update_plots_sonification()

    def select_full_triangle_rotated(self, event):
        self.selection_tool = "triangle_rotated"
        self.button_select_triangle_rotated.setChecked(True)
        if self.selected_area is not None:
            self.selected_area.remove()
        self.selected_area_start = [0, 0]
        self.selected_area_end = self.sonification_matrix.shape
        self.selected_area = self.plot_axes.add_patch(Polygon(self._triangle([0, 0], self.sonification_matrix.shape), closed=True, fill=False, edgecolor='cyan', linewidth=2))
        self.update_sonification_data()
        self.update_plots_sonification()

    def select_full_triangle_sheared(self, event):
        self.selection_tool = "triangle_sheared"
        self.button_select_triangle_sheared.setChecked(True)
        if self.selected_area is not None:
            self.selected_area.remove()
        self.selected_area_start = [0, 0]
        self.selected_area_end = self.sonification_matrix.shape
        self.selected_area = self.plot_axes.add_patch(Polygon(self._triangle([0, 0], self.sonification_matrix.shape), closed=True, fill=False, edgecolor='cyan', linewidth=2))
        self.update_sonification_data()
        self.update_plots_sonification()

    def update_plots_sonification(self):
        self.rec_plot.set_data(self.sonification_matrix)
        self.rec_plot.figure.canvas.draw_idle()
        self.sonification_img_1.set_data(self.sonification_data)
        self.sonification_img_2.set_data(np.rot90(self.sonification_data))
        self.fig_sonification.canvas.draw_idle()

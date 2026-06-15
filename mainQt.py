import sys

import numpy as np

import tsplayerQt as tsp

from matplotlib.backends.qt_compat import QtWidgets
from app import ApplicationWindow

if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

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

    time_series_players = [tsp.TSPlayer(ts) for ts in time_series]

    

    app = ApplicationWindow(time_series_players)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
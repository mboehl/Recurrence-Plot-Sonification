import sys

import numpy as np
from rpplayer import RPPlayer

from matplotlib.backends.qt_compat import QtWidgets
from app import ApplicationWindow

if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)


    # list of files to load as matrices
    files = ["data/climatic precession_distance_matrix.txt",
              "data/eccentricity_distance_matrix.txt",
                "data/periodic_distance_matrix.txt"]

    matrices = [np.loadtxt(file) for file in files]

    # create labels for each matrix based on the file name
    labels = [file.split("/")[-1].split(".")[0].split("_")[0] for file in files]
    
    # specify whether each matrix is a recurrence matrix or not
    is_rec_matrix = [False, False, False]  

    # create RPPlayer for each matrix
    recurrence_players = [RPPlayer(matrix, label=label, is_rec_matrix=is_rec) for matrix, label, is_rec in zip(matrices, labels, is_rec_matrix)]

    # create and show application window
    app = ApplicationWindow(recurrence_players)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
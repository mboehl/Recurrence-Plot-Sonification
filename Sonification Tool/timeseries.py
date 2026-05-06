import numpy as np
from pyunicorn.timeseries import RecurrencePlot

class TimeSeries():
    label : str
    time : np.array
    data : np.array
    fm_data: np.array

    def __init__(self, time, data, label, rec_downsample_factor=20, rec_cutoff=2000, rec_dim=5, rec_tau=5, rec_recurrence_rate=.05):
        self.time = time
        self.data = data
        self.label = label
        self.fm_data = self.frequency_modulation()
        self.rec_downsample_factor = rec_downsample_factor
        self.rec_cutoff = rec_cutoff
        self.rec_data = RecurrencePlot((self.data[::rec_downsample_factor])[:rec_cutoff], dim=rec_dim, tau=rec_tau, recurrence_rate=rec_recurrence_rate)

    def frequency_modulation(self, freq:int=220, sens:float=0.01, sample_rate=44100):
        """Returns freqency modulation of original time series with carrier frequency `f` and modulation sensitivity `sens`. The result is also saved as attribute `fm_data`."""
        time_stretch = 1
        self.fm_data = np.sin(np.linspace(0, 2 * np.pi * freq * len(self.time) / sample_rate * time_stretch, time_stretch * len(self.time)) + sens * np.cumsum(np.repeat(self.data, time_stretch, axis=0)))
        return self.fm_data[:len(self.time)]

    def recurrence_matrix(self):
        return self.rec_data.recurrence_matrix()


    def recurrence_extract_diagonal_lines(self):
        R = self.recurrence_matrix()
        N = R.shape[0]
        diag_line_centers = np.zeros((N, N), dtype=np.int16)

        # I counts steps along the main diagonal and J the steps in the
        # orthogonal direction. 
        for I in range(N):
            # do not include main diagonal
            main_diag = True
            streak = 0
            for J in range(min(I, N - I)):
                # i and j are the matrix indeces corresponding 
                # to the entry described by I and J.
                i, j = I - J, I + J

                if R[i][j] != 0:
                    streak += 1
                elif streak != 0:
                    if main_diag:
                        streak = 0
                        main_diag = False
                    else:
                        center = int(streak/2)
                        diag_line_centers[I][J - center] = streak
                        streak = 0
                              
        return diag_line_centers




if __name__ == "__main__":
    # M = np.array([[10*i + j for j in range(5)] for i in range(5)])
    # print(M)
    # print(half_diagonals_as_columns(M, dtype=np.int16))
    # import pase
    # pase.start()
    
    # fig, ax = plt.subplots(1, 2, layout="constrained")
    # ax[0].imshow(ts.recurrence_matrix(), origin="lower")
    # ax[1].imshow(half_diagonals_as_columns(ts.recurrence_matrix()), origin="lower")
    # plt.show()
    pass
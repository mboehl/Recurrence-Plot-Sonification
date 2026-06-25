from timeseries import TimeSeries
import numpy as np

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

time_series = [TimeSeries(time, series, label, rec_cutoff=1000, rec_downsample_factor=8, rec_dim=4, rec_tau=3) for series, label in zip(data[:-1], names[:-1])]
time_series.append(TimeSeries(time, np.sin(time), "periodic", rec_cutoff=500, rec_downsample_factor=2))

for ts in time_series:
    np.savetxt(f"data/{ts.label}_recurrence_matrix.txt", ts.recurrence_matrix())
    np.savetxt(f"data/{ts.label}_distance_matrix.txt", ts.distance_matrix())
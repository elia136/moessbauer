import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.style.use("./custom.mplstyle")

time, voltage = np.loadtxt("data/F0000CH1.CSV", delimiter=",", usecols=(3, 4), unpack=True)
time -= time[0]

fig, ax = plt.subplots()
ax.plot(time, voltage, linestyle="None", marker=".", markersize=4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
title = ax.set_title("Driver Unit Output Signal")
ax.grid(True)
plt.savefig("results/driver_unit_output.pdf", backend="pgf")

calibration = [
    {
        "file": "data/1700V_calib_energy.asc",
        "voltage": 1700,
        "label": "1700 V"
    },
    {
        "file": "data/1750V_calib_energy.asc",
        "voltage": 1750,
        "label": "1750 V"
    },
    {
        "file": "data/1800V_calib_energy.asc",
        "voltage": 1800,
        "label": "1800 V"
    },
    {
        "file": "data/1850V_calib_energy.asc",
        "voltage": 1850,
        "label": "1850 V"
    },
    {
        "file": "data/1900V_calib_energy.asc",
        "voltage": 1900,
        "label": "1900 V"
    },
    {
        "file": "data/1950V_calib_energy.asc",
        "voltage": 1950,
        "label": "1950 V"
    },
    {
        "file": "data/2000V_calib_energy.asc",
        "voltage": 2000,
        "label": "2000 V"
    },
    {
        "file": "data/2050V_calib_energy.asc",
        "voltage": 2050,
        "label": "2050 V"
    },
    {
        "file": "data/2100V_calib_energy.asc",
        "voltage": 2100,
        "label": "2100 V"
    }
]

for calib in calibration:
    counts = np.loadtxt(calib["file"])
    calib["counts"] = counts
    peak_idx = find_peaks(counts, height=100)
    calib["peak_pos"] = peak_idx


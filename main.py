import numpy as np
import pandas as pd
import plotly.graph_objects as plt
import plotly

from path import Path
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema

# set the plotly backend
pd.options.plotting.backend = "plotly"

# data preperation
def first_greater(series, threshold):
    return np.argmax(series > threshold)

def first_greater_abs(series, threshold):
    return np.argmax(np.abs(series) > threshold)

def cut_data(data, threshold, idx_offset):
    idx = first_greater_abs(data["CH1"], threshold) - idx_offset
    data_th = data[idx:idx+num_pts+idx_offset]

    return data_th

# data comparison
def area_size_comparison(data1, data2, idx_start, idx_end, dx):
    area1 = np.trapz(np.abs(data1[idx_start:idx_end]), dx=dx)
    area2 = np.trapz(np.abs(data2[idx_start:idx_end]), dx=dx)
    percent_area = np.abs(area1-area2) / area1 * 100
    return percent_area

def diff_area_comparison(data1, data2, idx_start, idx_end, dx):
    area1 = np.trapz(np.abs(data1[idx_start:idx_end]), dx=dx)
    diff_abs = np.abs(data1[idx_start:idx_end] - data2[idx_start:idx_end])
    percent_diff_area = np.sum(diff_abs) * dx / area1 * 100
    return percent_diff_area

def diff_zero_crossing(data1, data2, dx, num_crossings=10, plot_data=True):
    data1_zc = np.where(np.diff(np.sign(data1)))[0]
    data2_zc = np.where(np.diff(np.sign(data2)))[0]

    zc_diffs = np.abs(data1_zc[:num_crossings] - data2_zc[:num_crossings]) * dx

    if plot_data:
        off = 1
        num_values = 8
        freq1 = 2.0 / np.mean(np.diff(data1_zc[off:off+num_values]) * dx)
        freq2 = 2.0 / np.mean(np.diff(data2_zc[off:off+num_values]) * dx)

        plt.figure()
        plt.plot(data1)
        plt.plot(data2)
        plt.plot(data1_zc, [0] * len(data1_zc), 'g+')
        plt.plot(data2_zc, [0] * len(data2_zc), 'rx')
        plt.title(f"Zero Crossing Diffs: {zc_diffs}\nf1={freq1/1000:0.2f} kHz, f2={freq2/1000:0.2f} kHz")
    
    return zc_diffs


def compute(data_ref, data_test, label_ref, label_test, window_size):
    # filter the data
    data_ref["CH1"] = savgol_filter(data_ref["CH1"], savgol_window_size, savgol_order)
    data_test["CH1"] = savgol_filter(data_test["CH1"], savgol_window_size, savgol_order)


    ref_idx = first_greater_abs(data_ref["CH1"], threshold) - idx_offset
    data_ref_cut = cut_data(data_ref, threshold, idx_offset)
    data_test_cut = cut_data(data_test, threshold, idx_offset)

    # merge the data
    data = pd.DataFrame(data={
        "TIME": data_ref_cut["TIME"]*10e5,
        label_ref: data_ref_cut["CH1"],
        label_test: data_test_cut["CH1"]
    })

    # get the first peak
    idx_global_max = np.argmax(data[label_ref])

    # compute the error values
    idx_a = idx_global_max
    idx_b = idx_global_max + window_size

    area_size_value = area_size_comparison(data[label_ref], data[label_test], idx_a, idx_b, dx)
    diff_area_value = diff_area_comparison(data[label_ref], data[label_test], idx_a, idx_b, dx)

    # use the cut data without the visualization offset to avoid using zero crossing before the measurement starts
    data_ref_amp = cut_data(data_ref, threshold, 0)["CH1"].to_numpy()
    data_test_amp = cut_data(data_test, threshold, 0)["CH1"].to_numpy()
    
    diff_zero_crossings = diff_zero_crossing(data_ref_amp, data_test_amp, dx, num_zero_crossings, plot_data=False)

    print(f"Areas Size: {area_size_value}, Diff Area: {diff_area_value}, Diff Zero Crossings: {diff_zero_crossings}")

    # visualize
    fig = data.plot(x="TIME", y=[label_ref,label_test])
    
    fig.update_layout(
    shapes=[
        # 1st highlight during Feb 4 - Feb 6
        dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0=data["TIME"][ref_idx+idx_a],
            y0=0,
            x1=data["TIME"][ref_idx+idx_b],
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    ]
    )

    fig.update_layout(
        xaxis_title = "Time [\u03BCs]",
        yaxis_title ="Amplitude [V]",
        title = f"Areas Size comparison: {area_size_value:0.2f} %\nDifference Area comparison: {diff_area_value:0.2f} %\nDifference Zero Crossings [s]: {diff_zero_crossings}"
    )

    fig.show()


if __name__ == "__main__":
    # params
    threshold = 100         # min. absolute value for the thresholding
    idx_offset = 200        # offset to include data before the computed threshold value
    num_pts = 4000          # number of sampling points from the computed threshold value #default 7000
    dx = 32e-09             # sampling step size
    num_zero_crossings = 5  # number of zero crossing to output

    # set the window for the computed values
    window_size = 1250      #default 1500

    # filter settings (window size must be odd!)
    savgol_window_size = 31
    savgol_order = 3

    # change the name of the data legend
    label_ref = "Reference"
    label_test = "Test"


    # data loading => add your path to the files here
    base_dir = ""
    filepath_ref = Path(base_dir) / "T0001.csv"
    filepath_test = Path(base_dir) / "T0012.csv"

    data_ref = pd.read_csv(filepath_ref)
    data_test = pd.read_csv(filepath_test)

    # compute and plot
    compute(data_ref, data_test, label_ref, label_test, window_size)
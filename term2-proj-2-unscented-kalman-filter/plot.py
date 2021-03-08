import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple

UKFData = namedtuple('UKFData', "px,py,px_measured,py_measured,px_true,py_true")

def read_data(meas_number):
    output = pd.read_table("output-%d.txt" % meas_number)
    px = output['px']
    py = output['py']
    px_measured = output['px_measured']
    py_measured = output['py_measured']
    px_true = output['px_true']
    py_true = output['py_true']

    return UKFData(
        px, py,
        px_measured, py_measured,
        px_true, py_true)


def plot_data(num, ax):
    data = read_data(num)
    col = num - 1
    ax[col].set_title("sample {}".format(num))
    ax[col].plot(data.px_true, data.py_true, label="ground truth")
    ax[col].plot(data.px_measured, data.py_measured, ".", label="measurement")
    ax[col].plot(data.px, data.py, label="prediction")
    ax[col].legend(loc="lower right")


fig,ax = plt.subplots(1,2)
fig.set_size_inches(12, 6)
plot_data(1,ax)
plot_data(2,ax)

plt.savefig("result.png")

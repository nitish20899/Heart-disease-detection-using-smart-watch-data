import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import numpy as np
import os

path = './subject_4/watch'
tables = []
fnames = []

for root, dirs, files in os.walk(path):
    for file in files:   
        fname = os.path.join(path, file)
        try:
            csv = pd.read_csv(fname, delimiter='\t')
        except:
            print("Load Error")
            continue
        fnames.append(str(file))
        table = csv.to_numpy()
        tables.append(table[:, 1])

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
_ax = plt.axes([0.25, 0.15, 0.65, 0.03])

l, = ax.plot(tables[0])

slist = Slider(
    ax=_ax,
    label="index",
    valmin=0,
    valmax=len(tables) - 1,
    valstep=1,
    valinit=0
)


def update(val):
    z = slist.val
    table = tables[val]
    ax.set_title(fnames[val])
    print(fnames[val])
    ax.set_xlim([0, table.size])
    ax.set_ylim([np.min(table), np.max(table)])
    l.set_xdata(np.arange(0, table.size))
    l.set_ydata(table)


slist.on_changed(update)

plt.show()
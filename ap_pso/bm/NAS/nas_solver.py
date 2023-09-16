import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import CSS4_COLORS
from matplotlib.figure import Figure

input_file = Path("NAS2.out")  # The checkpoint file, where the data lies.
broken = True
if not broken:
    with open(input_file, "rb") as f:
        data = pickle.load(f, fix_imports=True)
        f.close()
    del input_file, f

else:
    with open(input_file) as f:
        text_data = f.read()
        f.close()
    del input_file, f

    # Everything before this is outside the program
    tdm = text_data.split("#-----------------------------------#\n| Current time: ")[1:]
    text_data = tdm[0]  # If there is a third section, it contains the end time and some additional trash.
    if len(tdm) > 1:
        end = tdm[1]
    else:
        end = ""

    start_time = int(text_data[:19])

    tdm = text_data.split("#-----------------------------------------#\n| [R")[1:]
    data = []
    for i, x in enumerate(tdm):
        rank, x = int(x[:2]), x[2:].split("] Current time: ")[1]
        time, x = int(x[:19]) / 1000000000, x[19:].split("Bred and evaluated individual [{'conv_layers': '")[1]
        loss = float(x[x.find("loss ") + 5:x.find(", island 0, worker ")])
        data.append(type("straw", (), {"g_rank": rank, "evaltime": time, "loss": loss}))
    del text_data, tdm, i, x, rank, time, loss, end

# =========================== Preparatory measures =========================== #

worker_colors = ("indianred", "tomato", "chocolate", "darkorange",
                 "darkgoldenrod", "olive", "forestgreen", "turquoise",
                 "darkcyan", "steelblue", "blue", "indigo",
                 "magenta", "crimson", "dimgray", "saddlebrown")
sorted_data = [(x.evaltime, x.loss, x.g_rank) for x in sorted(data, key=lambda y: y.evaltime) if x.loss <= 0]
time_data = [(x[0] - sorted_data[0][0]) / 3600 for x in sorted_data]
particle_scatter = {"x": time_data,
                    "y": [x[1] for x in sorted_data],
                    "c": [CSS4_COLORS[worker_colors[x[2] % 8]] for x in sorted_data]}

lsd = len(particle_scatter["x"])
tmp = ([np.median(particle_scatter["y"][:x]) for x in range(1, lsd + 1)],
       [np.min(particle_scatter["y"][:x]) for x in range(1, lsd + 1)])
median_data = time_data, tmp[0]
min_data = time_data, tmp[1]

# ========================= Start of graph painting ========================== #

fig: Figure
ax: Axes

fig, ax = plt.subplots()

ax.set_xlabel("Time (h)")
ax.set_ylabel("Loss value (negative accuracy)")
ax.set_ylim(-0.7, 0)

ax.scatter(**particle_scatter, s=4, label="Loss per particle, colorcoded by rank")
ax.plot(*median_data, color="r", label="Median loss")
ax.plot(*min_data, color="b", label="Minimum loss")

ax.legend()

fig.show()

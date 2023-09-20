"""
This file contains a second plotter program that uses some of the functionality of the graph plotter.
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ap_pso.bm.graph_plotter import create_time_data, core_counts, core_count_repr, pso_names, other_stuff, marker_list, \
    scaling_type

if __name__ == "__main__":
    time_data = create_time_data()
    averaged_time_data = {}

    for program in pso_names + other_stuff:
        data = np.array(list(time_data[program].values()))
        averaged_time_data[program] = [np.min(data, axis=0), np.average(data, axis=0), np.max(data, axis=0)]

    fig: Figure
    ax: Axes

    fig, ax = plt.subplots()

    ax.set_xlabel("Workers")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(core_counts), core_count_repr)
    ax.grid(True)
    if scaling_type == "strong":
        ax.set_ylabel("Speed-up")
    elif scaling_type == "weak":
        ax.set_ylabel("Efficiency")
    else:
        raise ValueError("Invalid scaling type.")

    for i, name in enumerate(pso_names + other_stuff):
        if i < 4:
            ms = 6
        else:
            ms = 7
        dp = np.array(averaged_time_data[name])
        for j, x in enumerate(dp):
            dp[j] = x[0] / x
        if scaling_type == "strong":
            # Show speed-up
            dp[0], dp[2] = dp[0] - dp[1], dp[1] - dp[2]
        elif scaling_type == "weak":
            # Show efficiency
            dp[0], dp[2] = dp[1] - dp[0], dp[2] - dp[1]
        else:
            raise ValueError("Invalid scaling type.")
        ax.errorbar(sorted(core_counts), dp[1],
                    [np.clip(dp[2], 0, None), np.clip(dp[0], 0, None)],
                    label=name, marker=marker_list[i], ms=ms)
    ax.legend()
    fig.show()

    save_path = Path(f"images/{scaling_type}_scaling.svg")
    if save_path.parent.exists() and not save_path.parent.is_dir():
        OSError("There is something in the way. We can't store our paintings.")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path.with_stem(save_path.stem + "_T"), transparent=True)
    fig.savefig(save_path.with_stem(save_path.stem + "_T").with_suffix(".pdf"), transparent=True)

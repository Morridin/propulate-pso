"""
This file contains the most-used graph plotter of them all.

This script is designed to plot graphs of every single benchmark function optimization I ran, for the weak scaling
series as well as for the strong scaling series.

Also, it handles some values for the overview plotter.
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

functions = (
    "Sphere",
    "Rosenbrock",
    "Step",
    "Quartic",
    "Rastrigin",
    "Griewank",
    "Schwefel",
    "BiSphere",
    "BiRastrigin",
)
pso_names = ("VelocityClamping", "Constriction", "Basic", "Canonical")
other_stuff = ("Vanilla Propulate", "Hyppopy")
marker_list = (
    "o",
    "s",
    "D",
    "^",
    "P",
    "X",
)  # ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D"]  # <-- Excess markers, in case you need...

# scaling_type = "strong"  # Select one of ...
scaling_type = "weak"  # ... these two lines.

if scaling_type == "strong":
    core_counts = [1, 2, 4, 8, 16]
    core_count_repr = [
        64,
        128,
        256,
        512,
        1024,
    ]  # [1, 2, 4, 8, 16]  # <-- The out commented array is the node count.
    time_path = Path("./slurm3/")
    path = Path("./results3/")
elif scaling_type == "weak":
    core_counts = [1, 2, 4, 8, 0.5]
    core_count_repr = [
        32,
        64,
        128,
        256,
        512,
    ]  # ["1/2", 1, 2, 4, 8]  # <-- The out commented array is the node count.
    time_path = Path("./slurm5A/")
    path = Path("./results5/")
else:
    raise ValueError("Invalid scaling type.")


def insert_data(d_array, idx: int, pt: Path):
    """
    This function adds the data given via `pt` into the data array given by `d_array` at position `idx`.
    """
    if not p.is_dir() or len([f for f in p.iterdir()]) == 0:
        return
    for fil in pt.iterdir():
        if not fil.suffix == ".pkl":
            continue
        with open(fil, "rb") as f:
            tm = pickle.load(f, fix_imports=True)
            d_array[idx].append(
                [
                    min(tm, key=lambda v: v.loss).loss,
                    (max(tm, key=lambda v: v.rank).rank + 1) / 64,
                ]
            )


def create_time_data() -> dict[str, dict[str, list[float]]]:
    """
    This function finds all necessary data to create all time values we need.

    Parameters
    ----------
    """

    def calc_time(iterator) -> float:
        """
        This function takes an iterator on a certain string array and calculates out of this a time span in seconds.
        """
        useless = "\n|: Ceirmnrtu"
        try:
            start = int(next(iterator).strip(useless))
        except ValueError:
            return np.nan
        try:
            end = int(next(iterator).strip(useless))
        except ValueError:
            return np.nan
        return (end - start) / 1e9

    raw_time_data: list[str] = []
    time_data: dict[str, dict[str, list[float]]] = {}

    for function_name in pso_names + other_stuff:
        time_data[function_name] = {}
        for program in functions:
            time_data[function_name][program] = []

    for file in time_path.iterdir():
        with open(file) as f:
            raw_time_data.append(f.read())

    for x in raw_time_data:
        scatter = [
            st
            for st in x.split("#-----------------------------------#")
            if "Current time" in st
        ]
        itx = iter(scatter)
        for program in other_stuff:
            for function_name in functions:
                time_data[program][function_name].append(calc_time(itx))
        for function_name in functions:
            for program in pso_names:
                time_data[program][function_name].append(calc_time(itx))
    return time_data


if __name__ == "__main__":
    time_data = create_time_data()

    for function_name in functions:
        data = []

        for i in range(5):
            data.append([])
            if i == 4:
                d = f"bm_P_{function_name.lower()}_?"
            else:
                d = f"bm_{i}_{function_name.lower()}_?"
            for p in path.glob(d):
                insert_data(data, i, p)
            data[i] = np.array(sorted(data[i], key=lambda v: v[1])).T
        data.append([])
        for p in path.glob(f"bm_H_{function_name.lower()}_?"):
            if not p.is_dir():
                continue
            file = p / Path("result_0.pkl")
            with open(file, "rb") as f:
                tmp = pickle.load(f, fix_imports=True)
                nodes = int(p.name[-1])
                data[-1].append([min(tmp[0]["losses"]), core_counts[nodes]])
        data[5] = np.array(sorted(data[5], key=lambda v: v[1])).T

        fig: Figure
        ax1: Axes
        ax2: Axes

        fig, (ax2, ax1) = plt.subplots(
            2, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": (3, 5)}
        )

        ax1.set_xlabel("Workers")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(sorted(core_counts), core_count_repr)
        ax1.grid(True)
        ax1.set_ylabel("Loss")

        ax2.grid(True)
        if scaling_type == "strong":
            ax2.set_ylabel("Speed-up")
        elif scaling_type == "weak":
            ax2.set_ylabel("Efficiency")
        else:
            raise ValueError("Unknown scaling type")
        ax2.set_yscale("log")
        everything = pso_names + other_stuff
        for i, name in enumerate(everything):
            if i < 4:
                ms = 6
            else:
                ms = 7
            ax1.plot(
                data[i][1],
                data[i][0],
                label=name,
                marker=marker_list[i],
                lw=0.75,
                ms=ms,
            )
            ax2.plot(
                data[i][1],
                time_data[name][function_name],
                marker=marker_list[i],
                lw=0.75,
                ms=ms,
            )

        if function_name == "Rosenbrock" and scaling_type == "strong":
            ax1.set_yscale("symlog", linthresh=1e-36)
            ax1.set_yticks([0, 1e-36, 1e-30, 1e-24, 1e-18, 1e-12, 1e-6, 1])
            ax1.set_ylim(-5e-36, 1)
        elif function_name == "Step":
            ax1.set_yscale("symlog")
            ax1.set_ylim(-1e5, -5)
        elif function_name == "Schwefel":
            ax1.set_yscale("symlog")
            ax1.set_ylim(-50000, 5000)
        elif (
            function_name in ("Schwefel", "Rastrigin", "BiSphere", "BiRastrigin")
            or function_name == "Rosenbrock"
            and scaling_type == "weak"
        ):
            ax1.set_yscale("linear")
        else:
            ax1.set_yscale("log")

        box = ax1.get_position(), ax2.get_position()
        ax1.set_position([box[0].x0, box[0].y0, box[0].width * 0.8, box[0].height])
        ax2.set_position([box[1].x0, box[1].y0, box[1].width * 0.8, box[1].height])

        fig.legend(loc="center right", bbox_to_anchor=(0.95, 0.5))
        fig.set_size_inches(10, 6)
        fig.show()

        save_path = Path(f"images/{scaling_type}/pso_{function_name.lower()}.svg")
        if save_path.parent.exists() and not save_path.parent.is_dir():
            raise OSError(
                "There is something in the way. We can't store our paintings."
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path.with_stem(save_path.stem + "_T"), transparent=True)
        fig.savefig(
            save_path.with_stem(save_path.stem + "_T").with_suffix(".pdf"),
            transparent=True,
        )

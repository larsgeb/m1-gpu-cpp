import numpy
import matplotlib.pyplot as plt
import pandas

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(4, 5))

plt.subplot(3, 1, 1)
results_laplacian = []
data_sizes = [16, 64, 256, 1024, 4096, 16384]
for data_size in data_sizes:
    results_laplacian.append(
        pandas.read_csv(
            f"./results/runtimes_function_{data_size}.csv",
            index_col=0,
            header=None,
        ).T
    )
categories = list(results_laplacian[0].columns.values)
dataframes = {}
for category in categories:
    dataframes[category] = {}
    for result, data_size in zip(results_laplacian, data_sizes):
        dataframes[category][data_size] = result[category]
labels = [
    "Metal (GPU)",
    "Serial",
    "OpenMP, 2 threads",
    "OpenMP, 4 threads",
    "OpenMP, 8 threads",
    "OpenMP, 10 threads",
]
medians = []
for i, category in enumerate(categories):
    medians.append([])
    color = colors[i]
    for data_size in data_sizes:
        data = dataframes[category][data_size]
        medians[i].append(numpy.median(dataframes[category][data_size]))
        if data.size > 1000:
            data = data.copy()[:: int(data.size / 1000)]
        plt.scatter(
            data_size * numpy.ones_like(data),
            data,
            alpha=1.0 / data.size**0.5,
            s=1,
            color=color,
        )

median_elemop_openmp8 = numpy.array(medians)[-1, -1]
median_elemop_serial = numpy.array(medians)[1, -1]
median_elemop_gpu = numpy.array(medians)[0, -1]

for i, category in enumerate(categories):
    color = colors[i]
    plt.plot(data_sizes, medians[i], color=color, label=labels[i], linewidth=0.8)
    plt.scatter(data_sizes, medians[i], color=color, s=1)
plt.grid()
plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.1)
plt.grid(visible=True, which="minor", color="k", linestyle="--", alpha=0.1)


plt.subplot(3, 1, 2)
results_laplacian = []
data_sizes = [16, 64, 256, 1024, 4096, 16384]
for data_size in data_sizes:
    results_laplacian.append(
        pandas.read_csv(
            f"./results/runtimes_laplacian_{data_size}.csv",
            index_col=0,
            header=None,
        ).T
    )
categories = list(results_laplacian[0].columns.values)
dataframes = {}
for category in categories:
    dataframes[category] = {}
    for result, data_size in zip(results_laplacian, data_sizes):
        dataframes[category][data_size] = result[category]
labels = [
    "Metal (GPU)",
    "Serial",
    "OpenMP, 2 threads",
    "OpenMP, 4 threads",
    "OpenMP, 8 threads",
    "OpenMP, 10 threads",
]
medians = []
for i, category in enumerate(categories):
    medians.append([])
    color = colors[i]
    for data_size in data_sizes:
        data = dataframes[category][data_size]
        medians[i].append(numpy.median(dataframes[category][data_size]))
        if data.size > 1000:
            data = data.copy()[:: int(data.size / 1000)]
        plt.scatter(
            data_size * numpy.ones_like(data),
            data,
            alpha=1.0 / data.size**0.5,
            s=1,
            color=color,
        )

median_lapl_openmp8 = numpy.array(medians)[-1, -1]
median_lapl_serial = numpy.array(medians)[1, -1]
median_lapl_gpu = numpy.array(medians)[0, -1]


for i, category in enumerate(categories):
    color = colors[i]
    plt.plot(data_sizes, medians[i], color=color, label=labels[i], linewidth=0.8)
    plt.scatter(data_sizes, medians[i], color=color, s=1)
plt.grid()
plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.1)
plt.grid(visible=True, which="minor", color="k", linestyle="--", alpha=0.1)


plt.subplot(3, 1, 3)
results_laplacian = []
data_sizes = [16, 64, 256, 1024, 4096, 16384]
for data_size in data_sizes:
    results_laplacian.append(
        pandas.read_csv(
            f"./results/runtimes_laplacian9p_{data_size}.csv",
            index_col=0,
            header=None,
        ).T
    )
categories = list(results_laplacian[0].columns.values)
dataframes = {}
for category in categories:
    dataframes[category] = {}
    for result, data_size in zip(results_laplacian, data_sizes):
        dataframes[category][data_size] = result[category]
labels = [
    "Metal (GPU)",
    "Serial",
    "OpenMP, 2 threads",
    "OpenMP, 4 threads",
    "OpenMP, 8 threads",
    "OpenMP, 10 threads",
]
medians = []
for i, category in enumerate(categories):
    medians.append([])
    color = colors[i]
    for data_size in data_sizes:
        data = dataframes[category][data_size]
        medians[i].append(numpy.median(dataframes[category][data_size]))
        if data.size > 1000:
            data = data.copy()[:: int(data.size / 1000)]
        plt.scatter(
            data_size * numpy.ones_like(data),
            data,
            alpha=1.0 / data.size**0.5,
            s=1,
            color=color,
        )
median_lapl9p_openmp8 = numpy.array(medians)[-1, -1]
median_lapl9p_serial = numpy.array(medians)[1, -1]
median_lapl9p_gpu = numpy.array(medians)[0, -1]

for i, category in enumerate(categories):
    color = colors[i]
    plt.plot(data_sizes, medians[i], color=color, label=labels[i], linewidth=0.8)
    plt.scatter(data_sizes, medians[i], color=color, s=1)
plt.grid()
plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.1)
plt.grid(visible=True, which="minor", color="k", linestyle="--", alpha=0.1)

plt.subplot(3, 1, 1)
plt.text(
    0.05,
    0.9,
    "Element-wise op",
    horizontalalignment="left",
    verticalalignment="center",
    fontweight="bold",
    transform=axes[0].transAxes,
)
plt.ylabel("runtime [ns]")

plt.subplot(3, 1, 2)
plt.text(
    0.05,
    0.9,
    "Laplacian 5-point",
    horizontalalignment="left",
    verticalalignment="center",
    fontweight="bold",
    transform=axes[1].transAxes,
)
plt.ylabel("runtime [ns]")

plt.subplot(3, 1, 3)
plt.text(
    0.05,
    0.9,
    "Laplacian 9-point",
    horizontalalignment="left",
    verticalalignment="center",
    fontweight="bold",
    transform=axes[2].transAxes,
)

plt.legend(loc="lower right", ncol=2, fontsize=5)

plt.yscale("log")
plt.xscale("log")
plt.gca().set_xticks(
    data_sizes,
    [
        f"$2^{{{int(numpy.log2(data_size))}}}$x$2^{{{int(numpy.log2(data_size))}}}$"
        for data_size in data_sizes
    ],
    rotation=15,
)
xlim = plt.xlim()
plt.gca().set_xticks(
    [2 ** float(i) for i in range(int(numpy.log2(data_sizes[-1])))],
    minor=True,
    labels=[],
)
plt.xlim(xlim)
plt.xlabel("data size")
plt.ylabel("runtime [ns]")
plt.ylim([1e1, 1e9])
plt.tight_layout()

fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("combined_02.png", dpi=300, format="png")
plt.close()

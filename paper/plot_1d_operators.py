import numpy
import matplotlib.pyplot as plt
import pandas

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(4, 3.75))

plt.subplot(2, 1, 1)
results_laplacian = []
data_sizes = [16, 128, 1024, 8192, 65536, 524288, 4194304, 33554432, 268435456]
for data_size in data_sizes:
    results_laplacian.append(
        pandas.read_csv(
            f"./results/runtimes_saxpy_{data_size}.csv",
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
        medians[i].append(numpy.mean(dataframes[category][data_size]))
        if data.size > 1000:
            data = data.copy()[:: int(data.size / 1000)]
        plt.scatter(
            data_size * numpy.ones_like(data),
            data,
            alpha=1.0 / data.size**0.5,
            s=1,
            color=color,
        )

saxpy_serial_speedup = (numpy.array(medians[0]) / numpy.array(medians[1]))[-1]
saxpy_OpenMP_speedup = (numpy.array(medians[0]) / numpy.array(medians[-2]))[-1]

for i, category in enumerate(categories):
    color = colors[i]
    plt.plot(data_sizes, medians[i], color=color, label=labels[i], linewidth=0.8)
    plt.scatter(data_sizes, medians[i], color=color, s=1)
plt.grid()
plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.1)
plt.grid(visible=True, which="minor", color="k", linestyle="--", alpha=0.1)


plt.text(
    0.05,
    0.9,
    "SAXPY",
    horizontalalignment="left",
    verticalalignment="center",
    fontweight="bold",
    transform=axes[0].transAxes,
)
plt.ylabel("runtime [ns]")

plt.subplot(2, 1, 2)


results_laplacian = []
data_sizes = [16, 128, 1024, 8192, 65536, 524288, 4194304, 33554432, 268435456]
for data_size in data_sizes:
    results_laplacian.append(
        pandas.read_csv(
            f"./results/runtimes_centraldif_{data_size}.csv",
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
        medians[i].append(numpy.mean(dataframes[category][data_size]))
        if data.size > 1000:
            data = data.copy()[:: int(data.size / 1000)]
        plt.scatter(
            data_size * numpy.ones_like(data),
            data,
            alpha=1.0 / data.size**0.5,
            s=1,
            color=color,
        )

cd_serial_speedup = (numpy.array(medians[0]) / numpy.array(medians[1]))[-1]
cd_OpenMP_speedup = (numpy.array(medians[0]) / numpy.array(medians[-2]))[-1]

for i, category in enumerate(categories):
    color = colors[i]
    plt.plot(data_sizes, medians[i], color=color, label=labels[i], linewidth=0.8)
    plt.scatter(data_sizes, medians[i], color=color, s=1)
plt.grid()
plt.grid(visible=True, which="major", color="k", linestyle="-", alpha=0.1)
plt.grid(visible=True, which="minor", color="k", linestyle="--", alpha=0.1)


plt.text(
    0.05,
    0.9,
    "Central differencing 3-point",
    horizontalalignment="left",
    verticalalignment="center",
    fontweight="bold",
    transform=axes[1].transAxes,
)
plt.ylabel("runtime [ns]")

plt.subplot(2, 1, 1)

plt.yscale("log")
plt.xscale("log")


plt.subplot(2, 1, 2)
plt.legend(loc="lower right", ncol=2, fontsize=5)

plt.yscale("log")
plt.xscale("log")
plt.gca().set_xticks(
    data_sizes,
    [f"2^{int(numpy.log2(data_size))}" for data_size in data_sizes],
    rotation=25,
)

xlim = plt.xlim()
plt.gca().set_xticks(
    [2 ** float(i) for i in range(int(numpy.log2(data_sizes[-1])))],
    minor=True,
    labels=[],
)
plt.gca().set_xticks(
    data_sizes,
    [f"$2^{{{int(numpy.log2(data_size))}}}$" for data_size in data_sizes],
    rotation=0,
)

plt.xlim(xlim)

plt.xlabel("data size")
plt.ylabel("runtime [ns]")

# plt.ylim([1e1, 1e8])
plt.tight_layout()

fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("combined_01.png", dpi=300, format="png")
plt.close()

print(f"cd speedup w.r.t. serial: {cd_serial_speedup**-1:.1f}")
print(f"saxpy speedup w.r.t. serial: {saxpy_serial_speedup**-1:.1f}")

print(f"cd speedup w.r.t. OpenMP: {cd_OpenMP_speedup**-1:.1f}")
print(f"saxpy speedup w.r.t. OpenMP: {saxpy_OpenMP_speedup**-1:.1f}")

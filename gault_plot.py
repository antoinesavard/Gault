import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import data as dtf

# install xarray first
import xarray as xr

# data dependent values
# ---------------------------------------------------------------
time_index = 0
zoom_start = 7
zoom_end = -2
buoy_index = -3
# ---------------------------------------------------------------

# structure
# data array
da_ice = xr.DataArray(
    np.nan_to_num(dtf.ice_array),
    dims=("time", "distance"),
    coords={
        "time": ("time", pd.to_datetime(dtf.date), {"type": "YYYY-MM-DD"}),
        "distance": (["time", "distance"], dtf.distance_array, {"units": "meters"}),
    },
    attrs=dict(
        description="From base of the ice", units="centimeters", Fillvalue=np.NaN
    ),
    name="ice thickness",
)
da_slush = xr.DataArray(
    np.nan_to_num(dtf.slush_array - dtf.ice_array),
    dims=("time", "distance"),
    coords={
        "time": ("time", pd.to_datetime(dtf.date), {"type": "YYYY-MM-DD"}),
        "distance": (["time", "distance"], dtf.distance_array, {"units": "meters"}),
    },
    attrs=dict(
        description="From base of the ice", units="centimeters", Fillvalue=np.NaN
    ),
    name="slush thickness",
)

da_snow = xr.DataArray(
    dtf.snow_array
    - np.nan_to_num(
        np.where(np.isnan(dtf.slush_array), dtf.ice_array, dtf.slush_array)
    ),
    dims=("time", "distance"),
    coords={
        "time": ("time", pd.to_datetime(dtf.date), {"type": "YYYY-MM-DD"}),
        "distance": (["time", "distance"], dtf.distance_array, {"units": "meters"}),
    },
    attrs=dict(
        description="From base of the ice", units="centimeters", Fillvalue=np.NaN
    ),
    name="snow thickness",
)

# dictionary
dic = {
    "ice_thickness": da_ice,
    "slush_thickness": da_slush,
    "snow_thickness": da_snow,
}

# dataset
xds = xr.Dataset(dic)

# Extract x values and variables
distance = xds.distance.values
distance_labels = distance[time_index].astype(str)
distance = np.arange(len(distance[time_index]))
ice = xds["ice_thickness"].values
slush = xds["slush_thickness"].values
snow = xds["snow_thickness"].values

# Create stacked bar plots
fig, ax = plt.subplots()
fig.set_tight_layout(True)

# aspect ratio for buoy
pos = ax.get_position()
width = pos.width
height = pos.height
aspect_ratio = (
    (-np.min(-snow[time_index] - slush[time_index] - ice[time_index]) + 8)
    / np.max(distance)
    / (height / width)
)

# plot buoy
radius = 0.5
circle = Ellipse(
    (distance[buoy_index], 1),
    radius,
    radius * aspect_ratio,
    color="orange",
    label="buoy",
)
ax.add_patch(circle)

# plot
ax.stackplot(
    distance,
    -snow[time_index],
    -slush[time_index],
    -ice[time_index],
    colors=["xkcd:baby blue", "xkcd:dark blue", "xkcd:bright blue"],
    labels=["snow", "slush", "ice"],
)

# Labels and title and legend
ax.set_xlabel("Distance [m]")
ax.set_xticks(distance[::2])
ax.set_xticklabels(distance_labels[::2])
ax.set_ylabel(
    "Thickness [cm]",
    rotation=0,
    multialignment="left",
    ha="right",
    position=(0, 0.9),
)
ax.set_title(xds.coords["time"].dt.strftime("%Y-%m-%d").values[time_index])
ax.set_ylim((np.min(-snow[time_index] - slush[time_index] - ice[time_index]) - 5, 3))
ax.set_xlim((0, np.max(distance)))

# vertical lines
ylenght = ax.get_ylim()[0]
ax.plot(
    np.ones(10) * distance[zoom_start],
    np.linspace(
        -snow[time_index, zoom_start],
        ylenght,
        10,
    ),
    "k:",
    label=r"$10\>$m gap",
)
ax.plot(
    np.ones(10) * distance[zoom_end],
    np.linspace(
        -snow[time_index, zoom_end],
        ylenght,
        10,
    ),
    "k:",
)
ax.set_ylim(ylenght)

# legend
ax.legend(
    loc="lower right",
    bbox_to_anchor=(-0.1, 0.4),
    frameon=False,
    ncol=1,
    handlelength=0.8,
)

# Show plot
plt.savefig(
    "plots/gault_" + xds.coords["time"].dt.strftime("%Y-%m-%d").values[0] + ".pdf"
)

# Save to netCDF
xds.to_netcdf("gault.nc")

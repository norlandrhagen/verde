"""
Spatial splitting for cross-validation
======================================

Evaluating a gridder's performance requires having a separate dataset to compare to our
predictions. Verde provides ways of splitting a dataset into one for fitting the gridder
(a training set) and one for comparing to predictions (a testing set). Function
:func:`verde.train_test_split` is based on
:func:`sklearn.model_selection.train_test_split` but is able to handle spatial data as
inputs.

See :ref:`model_evaluation` for more details.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyproj
import numpy as np
import verde as vd

# Load the Baja California shipborne bathymetry data
data = vd.datasets.fetch_baja_bathymetry()
coordinates = (data.longitude.values, data.latitude.values)
region = vd.get_region(coordinates)

# Split the data into a training and testing set by picking points at random
# This is NOT the best way to split spatially correlated data and will cease being the
# default in future versions of Verde.
train, test = vd.train_test_split(coordinates, data.bathymetry_m, random_state=2)
print(train[0][0].size, test[0][0].size)

# Alternatively, we can split the data into blocks and pick blocks at random.
# The advantage of this approach is that it makes sure that the training and testing
# datasets are not spatially correlated, which would bias our model evaluation.
# This will be the default in future versions of Verde.
block_train, block_test = vd.train_test_split(
    coordinates, data.bathymetry_m, method="block", spacing=1, random_state=0,
)

print(block_train[0][0].size, block_test[0][0].size)

fig, axes = plt.subplots(
    1, 2, figsize=(10, 6), subplot_kw=dict(projection=ccrs.Mercator())
)
crs = ccrs.PlateCarree()
ax = axes[0]
ax.set_title("Shuffle Split")
ax.scatter(*train[0], c="blue", s=10, transform=crs, label="train")
ax.scatter(*test[0], c="red", s=10, transform=crs, label="test")
vd.datasets.setup_baja_bathymetry_map(ax)
ax.coastlines()
ax.legend(loc="upper right")
ax = axes[1]
ax.set_title("Block Shuffle Split")
ax.scatter(*block_train[0], c="blue", s=10, transform=crs, label="train")
ax.scatter(*block_test[0], c="red", s=10, transform=crs, label="test")
vd.datasets.setup_baja_bathymetry_map(ax)
ax.coastlines()
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

"""
.. _model_selection:

Model Selection
===============

In :ref:`model_evaluation`, we saw how to check the performance of an interpolator using
cross-validation. We found that the default parameters for :class:`verde.Spline` are not
good for predicting our sample air temperature data. Now, let's see how we can tune the
:class:`~verde.Spline` to improve the cross-validation performance.

Once again, we'll start by importing the required packages.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import itertools
import pyproj
import verde as vd

########################################################################################
# And loading, projecting, and decimating our sample bathymetry data.
data = vd.datasets.fetch_baja_bathymetry()

# Use Mercator projection because Spline is a Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())

# Decimate to the desired grid spacing.
spacing = 10 / 60
reducer = vd.BlockReduce(reduction=np.median, spacing=spacing)
coordinates, bathymetry = reducer.filter(
    (data.longitude, data.latitude), data.bathymetry_m
)
proj_coords = projection(*coordinates)
region = vd.get_region(coordinates)

########################################################################################
# Before we begin tuning, let's reiterate what the results were with the default
# parameters.

# We'll use the spatially blocked version of k-fold cross-validation
kfold = vd.BlockKFold(spacing=1 * 111000, shuffle=True, random_state=0)

spline_default = vd.Spline()
score_default = np.mean(
    vd.cross_val_score(spline_default, proj_coords, bathymetry, cv=kfold)
)
spline_default.fit(proj_coords, bathymetry)
print("R² with defaults:", score_default)


########################################################################################
# Tuning
# ------
#
# :class:`~verde.Spline` has many parameters that can be set to modify the final result.
# Mainly the ``damping`` regularization parameter and the ``mindist`` "fudge factor"
# which smooths the solution. Would changing the default values give us a better score?
#
# We can answer these questions by changing the values in our ``spline`` and
# re-evaluating the model score repeatedly for different values of these parameters.
# Let's test the following combinations:

# We'll use few combinations for the sake of time
dampings = [None, 1e-5]
mindists = [5e3, 50e3]

# Use itertools to create a list with all combinations of parameters to test
parameter_sets = [
    dict(damping=combo[0], mindist=combo[1])
    for combo in itertools.product(dampings, mindists)
]
print("Number of combinations:", len(parameter_sets))
print("Combinations:", parameter_sets)

########################################################################################
# Now we can loop over the combinations and collect the scores for each parameter set.

spline = vd.Spline()
scores = []
for params in parameter_sets:
    spline.set_params(**params)
    score = np.mean(vd.cross_val_score(spline, proj_coords, bathymetry, cv=kfold))
    scores.append(score)
print(scores)

########################################################################################
# The largest score will yield the best parameter combination.

best = np.argmax(scores)
print("Best score:", scores[best])
print("Score with defaults:", score_default)
print("Best parameters:", parameter_sets[best])

########################################################################################
# That is a decent improvement over our previous score.
#
# This type of tuning is important and should always be performed when using a new
# gridder or a new dataset. However, the above implementation requires a lot of
# coding. Fortunately, Verde provides convenience classes that perform the
# cross-validation and tuning automatically when fitting a dataset.


########################################################################################
# Cross-validated gridders
# ------------------------
#
# The :class:`verde.SplineCV` class provides a cross-validated version of
# :class:`verde.Spline`. It has almost the same interface but does all of the above
# automatically when fitting a dataset. The only difference is that you must provide a
# list of ``damping`` and ``mindist`` parameters to try instead of only a single value:

spline = vd.SplineCV(dampings=[None, 1e-5], mindists=[5e3, 50e3], cv=kfold)
spline.fit(proj_coords, bathymetry)

########################################################################################
# The estimated best damping and mindist, as well as the cross-validation scores, are
# stored in class attributes:

print("Highest score:", spline.scores_.max())
print("Best damping:", spline.damping_)
print("Best mindist:", spline.mindist_)

########################################################################################
# Finally, we can make a grid with the best configuration to see how it compares to the
# default result.

grid = spline.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names=["bathymetry"],
)
print(grid)

########################################################################################
# Let's plot our grid side-by-side with one generated by the default spline:

grid_default = spline_default.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names=["bathymetry"],
)

mask = vd.distance_mask(
    (data.longitude, data.latitude),
    maxdist=3 * spacing * 111e3,
    coordinates=vd.grid_coordinates(region, spacing=spacing),
    projection=projection,
)

grid = grid.where(mask)
grid_default = grid_default.where(mask)

plt.figure(figsize=(14, 10))
for i, title, grd in zip(range(2), ["Defaults", "Tuned"], [grid_default, grid]):
    ax = plt.subplot(1, 2, i + 1, projection=ccrs.Mercator())
    ax.set_title(title)
    pc = grd.bathymetry.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=bathymetry.min(),
        vmax=bathymetry.max(),
        add_colorbar=False,
        add_labels=False,
    )
    plt.colorbar(pc, orientation="horizontal", aspect=50, pad=0.05).set_label("C")
    vd.datasets.setup_baja_bathymetry_map(ax, land=None)
plt.tight_layout()
plt.show()

########################################################################################
# Notice that **smoother models tend to be better predictors**. This is a sign that you
# should probably not trust many of the short wavelength features that we get from the
# defaults.

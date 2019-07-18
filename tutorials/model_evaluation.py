"""
.. _model_evaluation:

Evaluating Performance
======================

The Green's functions based interpolations in Verde are all linear regressions under the
hood. This means that we can use some of the same tactics from
:mod:`sklearn.model_selection` to evaluate our interpolator's performance. Once we have
a quantified measure of the quality of a given fitted gridder, we can use it to tune the
gridder's parameters, like ``damping`` for a :class:`~verde.Spline` (see
:ref:`model_selection`).
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyproj
import verde as vd

########################################################################################
# Verde provides adaptations of common scikit-learn tools to work better with spatial
# data. Let's use these tools to evaluate the performance of a :class:`~verde.Spline` on
# our sample bathymetry data.

data = vd.datasets.fetch_baja_bathymetry()

# Use Mercator projection because Spline is a Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())

########################################################################################
# Before gridding, we need to decimate the data to avoid aliasing because of the
# oversampling along the ship tracks. We'll use a blocked median with 10 arc-minute
# blocks since that is our desired grid spacing.
spacing = 10 / 60
reducer = vd.BlockReduce(reduction=np.median, spacing=spacing)
coordinates, bathymetry = reducer.filter(
    (data.longitude, data.latitude), data.bathymetry_m
)
proj_coords = projection(*coordinates)
region = vd.get_region(coordinates)

########################################################################################
# Splitting the data randomly
# ---------------------------
#
# We can't evaluate a gridder on the data that went into fitting it. The true test of a
# model is if it can correctly predict data that it hasn't seen before. scikit-learn has
# the :func:`sklearn.model_selection.train_test_split` function to separate a dataset
# into two parts: one for fitting the model (called *training* data) and a separate one
# for evaluating the model (called *testing* data). Using it with spatial data would
# involve some tedious array conversions so Verde implements
# :func:`verde.train_test_split` which does the same thing but takes coordinates and
# data arrays instead.
#
# The split is done randomly so we specify a seed for the random number generator to
# guarantee that we'll get the same result every time we run this example. You probably
# don't want to do that for real data. We'll keep 30% of the data to use for testing
# (``test_size=0.3``).

train, test = vd.train_test_split(
    proj_coords, bathymetry, test_size=0.3, random_state=0,
)
# The test size should roughly 30% of the available data
print(train[0][0].size, test[0][0].size)

########################################################################################
# The returned ``train`` and ``test`` variables are tuples containing coordinates, data,
# and (optionally) weights arrays. Since we're not using weights, the third element of
# the tuple will be ``None``:

print(train)

########################################################################################
# Let's plot the points belonging two datasets with different colors:

plt.figure(figsize=(8, 6))
ax = plt.axes()
ax.plot(train[0][0], train[0][1], ".r", label="train")
ax.plot(test[0][0], test[0][1], ".b", label="test")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()


########################################################################################
# Scoring
# --------
#
# Gridders in Verde implement the :meth:`~verde.base.BaseGridder.score` method that
# calculates the `R² coefficient of determination
# <https://en.wikipedia.org/wiki/Coefficient_of_determination>`__
# for a given comparison dataset (``test`` in our case). The R² score is at most 1,
# meaning a perfect prediction, but has no lower bound.
#
# We can pass the training dataset to the :meth:`~verde.base.BaseGridder.fit` method of
# most gridders using Python's argument expansion using the ``*`` symbol. The same can
# be applied to the testing set:

spline = vd.Spline()
spline.fit(*train)
score = spline.score(*test)
print("R² score:", score)

########################################################################################
# That's a good score meaning that our gridder is able to accurately predict data that
# wasn't used in the gridding algorithm.
#
# .. caution::
#
#     Once caveat for this score is that it is dependent on the particular split that we
#     made. Changing the random number generator seed in :func:`verde.train_test_split`
#     will result in a different score.

train_other, test_other = vd.train_test_split(
    proj_coords, bathymetry, test_size=0.3, random_state=2,
)

print("R² score different random state:", vd.Spline().fit(*train_other).score(*test_other))

########################################################################################
# .. caution::
#
#     Another caveat is that our data are *spatially correlated* (near by points tend to
#     have similar values). Having testing points close to training points will tend to
#     inflate the score. Splitting the data using blocks leads to a more honest score
#     [Roberts2017]_.


########################################################################################
# Splitting the data in blocks
# ----------------------------
#
# [Roberts2017]_

train, test = vd.train_test_split(
    proj_coords, bathymetry, test_size=0.3, random_state=0, method="block",
    spacing=1*111000
)
# The test size should roughly 30% of the available data
print(train[0][0].size, test[0][0].size)

plt.figure(figsize=(8, 6))
ax = plt.axes()
ax.plot(train[0][0], train[0][1], ".r", label="train")
ax.plot(test[0][0], test[0][1], ".b", label="test")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

spline = vd.Spline()
spline.fit(*train)
score = spline.score(*test)
print("R² score:", score)

train_other, test_other = vd.train_test_split(
    proj_coords, bathymetry, test_size=0.3, random_state=10, method="block",
    spacing=1*111000
)

print("R² score different random state:", vd.Spline().fit(*train_other).score(*test_other))


########################################################################################
# Cross-validation
# ----------------
#
# A more robust way of scoring the gridders is to use function
# :func:`verde.cross_val_score`, which uses `k-fold cross-validation
# <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`__
# by default. It will split the data *k* times and return the score on each *fold*. We
# can then take a mean of these scores.

scores = vd.cross_val_score(vd.Spline(), proj_coords, bathymetry)
print("k-fold scores:", scores)
print("Mean score:", np.mean(scores))

########################################################################################
# You can also use most cross-validation splitter classes from
# :mod:`sklearn.model_selection` and Verde by specifying the ``cv`` argument.
#
#
# example, if we want to shuffle then split the data blocks *n* times
# (:class:`verde.BlockShuffleSplit`):

shuffle = vd.BlockShuffleSplit(n_splits=10, test_size=0.3, random_state=0,
                               spacing=1*111000)

scores = vd.cross_val_score(vd.Spline(), proj_coords, bathymetry, cv=shuffle)
print("shuffle scores:", scores)
print("Mean score:", np.mean(scores))

########################################################################################
# That is not a bad score but we can do better than using the default arguments for
# :class:`~verde.Spline`. We could try different combinations manually until we get a
# good score. A better way is to do this automatically. In :ref:`model_selection` we'll
# go over how to do that.

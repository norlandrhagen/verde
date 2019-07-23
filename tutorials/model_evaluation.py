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
    proj_coords, bathymetry, test_size=0.3, random_state=0
)
# The test size should be roughly 30% of the available data
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
#     **Caveat**: for this score is that it is dependent on the particular split that we
#     made. Changing the random number generator seed in :func:`verde.train_test_split`
#     will result in a different score.

train_other, test_other = vd.train_test_split(
    proj_coords, bathymetry, test_size=0.3, random_state=2
)

print(
    "R² score different random state:", vd.Spline().fit(*train_other).score(*test_other)
)

########################################################################################
# That score isn't too different but this effect can be much larger for smaller
# datasets.

########################################################################################
# Splitting the data in blocks
# ----------------------------
#
# One thing to note is that our data are *spatially correlated* (nearby points tend to
# have similar values). Having testing points close to training points will tend to
# inflate the score. Splitting the data using blocks leads to a more honest score
# [Roberts2017]_. We can do this with :func:`verde.train_test_split` by specifying the
# *block* method of splitting.

train, test = vd.train_test_split(
    proj_coords,
    bathymetry,
    test_size=0.3,
    random_state=0,
    method="block",
    spacing=1 * 111000,
)

# The test size should still be roughly 30% of the available data
print(train[0][0].size, test[0][0].size)

########################################################################################
# The data gets grouped into blocks (with size specified by ``spacing``) and the blocks
# get split into training and testing. We can see this clearly when we visualize the
# split:

plt.figure(figsize=(8, 6))
ax = plt.axes()
ax.plot(train[0][0], train[0][1], ".r", label="train")
ax.plot(test[0][0], test[0][1], ".b", label="test")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

########################################################################################
# Training and scoring our
spline = vd.Spline()
spline.fit(*train)
score = spline.score(*test)
print("R² score:", score)

train_other, test_other = vd.train_test_split(
    proj_coords,
    bathymetry,
    test_size=0.3,
    random_state=10,
    method="block",
    spacing=1 * 111000,
)

print(
    "R² score different random state:", vd.Spline().fit(*train_other).score(*test_other)
)

########################################################################################
# Again we see that changing the random state leads to very different scores.
#
# Cross-validation
# ----------------
#
# A more robust way of scoring the gridders is to use function
# :func:`verde.cross_val_score`, which uses `k-fold cross-validation
# <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`__
# through :class:`sklearn.model_selection.KFold` by default. It will split the data *k*
# times and return the score on each *fold*. We can then take a mean of these scores.
# By default, the data is shuffled prior to splitting.

scores = vd.cross_val_score(vd.Spline(), proj_coords, bathymetry)
print("k-fold scores:", scores)
print("Mean score:", np.mean(scores))

########################################################################################
# You can also use most cross-validation splitter classes from
# :mod:`sklearn.model_selection` and Verde by specifying the ``cv`` argument.
#
# As we've seen before, randomly splitting the data can lead to inflated scores. Verde
# offers a spatially blocked version of k-fold through :class:`verde.BlockKFold`:

kfold = vd.BlockKFold(n_splits=5, shuffle=True, random_state=0, spacing=1 * 111000)
scores = vd.cross_val_score(vd.Spline(), proj_coords, bathymetry, cv=kfold)
print("block k-fold scores:", scores)
print("Mean score:", np.mean(scores))

########################################################################################
# That is not a bad score but we can do better than using the default arguments for
# :class:`~verde.Spline`. We could try different combinations manually until we get a
# good score. A better way is to do this automatically. In :ref:`model_selection` we'll
# go over how to do that.

########################################################################################
# Visualizing blocked k-fold
# --------------------------
#
# It's easier to understand how k-fold works by visualizing each of the folds. First,
# lets plot the train and test sets for a non-randomized blocked k-fold:

kfold = vd.BlockKFold(n_splits=4, shuffle=False, spacing=1 * 111000)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
for i, (train_index, test_index) in enumerate(kfold.split(proj_coords)):
    ax = axes.ravel()[i]
    ax.plot(
        proj_coords[0][train_index], proj_coords[1][train_index], ".r", label="train"
    )
    ax.plot(proj_coords[0][test_index], proj_coords[1][test_index], ".b", label="test")
    ax.set_aspect("equal")
    ax.set_title("Fold {}".format(i + 1))
ax.legend()
plt.tight_layout()
plt.show()

########################################################################################
# As the figure shows, non-random folds are spatially grouped. Any gridder would have a
# difficult time accurately predicting the test data in this situation. For this reason,
# it is better to shuffle the blocks:

kfold = vd.BlockKFold(n_splits=4, shuffle=True, spacing=1 * 111000, random_state=0)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
for i, (train_index, test_index) in enumerate(kfold.split(proj_coords)):
    ax = axes.ravel()[i]
    ax.plot(
        proj_coords[0][train_index], proj_coords[1][train_index], ".r", label="train"
    )
    ax.plot(proj_coords[0][test_index], proj_coords[1][test_index], ".b", label="test")
    ax.set_aspect("equal")
    ax.set_title("Fold {}".format(i + 1))
ax.legend()
plt.tight_layout()
plt.show()

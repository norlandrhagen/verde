"""
Functions for automating model selection through cross-validation.

Supports using a dask.distributed.Client object for parallelism. The
DummyClient is used as a serial version of the parallel client.
"""
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, BaseCrossValidator

from .base import check_fit_input
from .utils import DummyClient
from .coordinates import block_split


class BaseBlockCrossValidator(BaseCrossValidator):
    """
    Base class for spatially blocked cross-validators.

    Instead of splitting the data randomly or in folds, divide the data into spatial
    blocks and split the blocks between train and test. See [Roberts2017]_.
    """

    is_spatial = True

    def __init__(
        self,
        n_splits,
        test_size=0.1,
        train_size=None,
        random_state=None,
        region=None,
        spacing=None,
        shape=None,
        adjust="spacing",
        iterations=50,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.region = region
        self.spacing = spacing
        self.shape = shape
        self.adjust = adjust
        self.iterations = iterations

    def get_n_splits(self):
        """
        """
        return self.n_splits

    def split(self, coordinates):
        """
        """
        _, labels = block_split(
            coordinates,
            spacing=self.spacing,
            shape=self.shape,
            region=self.region,
            adjust=self.adjust,
        )
        block_ids = np.unique(labels)
        shuffle = ShuffleSplit(
            n_splits=self.n_splits * self.iterations,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        ).split(block_ids)
        for _ in range(self.n_splits):
            trains, tests, balance = [], [], []
            for _ in range(self.iterations):
                train_id, test_id = next(shuffle)
                trains.append(np.where(np.isin(labels, block_ids[train_id]))[0])
                tests.append(np.where(np.isin(labels, block_ids[test_id]))[0])
                balance.append(
                    abs(trains[-1].size / tests[-1].size - train_id.size / test_id.size)
                )
            best = np.argmin(balance)
            yield trains[best], tests[best]


class BlockShuffleSplit(BaseBlockCrossValidator):
    """
    Random spatial block permutation cross-validator.
    """

    def __init__(
        self,
        n_splits=10,
        test_size=0.1,
        train_size=None,
        random_state=None,
        region=None,
        spacing=None,
        shape=None,
        adjust="spacing",
        iterations=50,
    ):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size,
              random_state=random_state, region=region, spacing=spacing, shape=shape,
              adjust=adjust, iterations=iterations,)

    def split(self, coordinates):
        """
        """
        _, labels = block_split(
            coordinates,
            spacing=self.spacing,
            shape=self.shape,
            region=self.region,
            adjust=self.adjust,
        )
        block_ids = np.unique(labels)
        shuffle = ShuffleSplit(
            n_splits=self.n_splits * self.iterations,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        ).split(block_ids)
        for _ in range(self.n_splits):
            trains, tests, balance = [], [], []
            for _ in range(self.iterations):
                train_id, test_id = next(shuffle)
                trains.append(np.where(np.isin(labels, block_ids[train_id]))[0])
                tests.append(np.where(np.isin(labels, block_ids[test_id]))[0])
                balance.append(
                    abs(trains[-1].size / tests[-1].size - train_id.size / test_id.size)
                )
            best = np.argmin(balance)
            yield trains[best], tests[best]


def train_test_split(coordinates, data, weights=None, method="random", **kwargs):
    r"""
    Split a dataset into a training and a testing set for cross-validation.

    Similar to :func:`sklearn.model_selection.train_test_split` but is tuned to
    work on multi-component spatial data with optional weights.

    Extra keyword arguments will be passed to
    :class:`sklearn.model_selection.ShuffleSplit` or :class:`verde.BlockShuffleSplit`
    (depending on *method*), except for ``n_splits`` which is always 1.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...).
    data : array or tuple of arrays
        the data values of each data point. If the data has more than one
        component, *data* must be a tuple of arrays (one for each component).
    weights : None or array or tuple of arrays
        if not none, then the weights assigned to each data point. If more than
        one data component is provided, you must provide a weights array for
        each data component (if not none).
    method : str
        The method used to split the data. Can be either ``"random"`` meaning a random
        split (:class:`sklearn.model_selection.ShuffleSplit`) or ``"block"`` meaning a
        random split using spatial blocks (:class:`verde.BlockShuffleSplit`). If method
        *block* is used, a ``spacing`` or ``shape`` argument **must be provided** to
        specify the size or number, respectively, of blocks.
    **kwargs
        Extra arguments are passed to the cross-validation class used depending on the
        chosen *method*.

    Returns
    -------
    train, test : tuples
        Each is a tuple = (coordinates, data, weights) with the data corresponding to
        each set.

    Examples
    --------

    The random splitting mode will divide the data into two sets ignoring the spatial
    distribution of the data:

    >>> import numpy as np
    >>> data = np.array([1, 3, 5, 7])
    >>> coordinates = (np.arange(4), np.arange(-4, 0))
    >>> # Control the random state so we get reproducible results
    >>> train, test = train_test_split(coordinates, data, random_state=0)
    >>> # Show the coordinates and data values belonging to each set
    >>> print("Coordinates:", train[0])
    Coordinates: (array([3, 1, 0]), array([-1, -3, -4]))
    >>> print("Data:", train[1])
    Data: (array([7, 3, 1]),)
    >>> print("Weights:", train[2])
    Weights: (None,)
    >>> print("Coordinates:", test[0])
    Coordinates: (array([2]), array([-2]))
    >>> print("Data:", test[1])
    Data: (array([5]),)
    >>> print("Weights:", test[2])
    Weights: (None,)

    Data with more than 1 component and/or weights can also be handled:

    >>> data = (np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))
    >>> weights = (np.array([1, 1, 2, 1]), np.array([1, 2, 1, 1]))
    >>> train, test = train_test_split(
    ...     coordinates, data, weights, random_state=0
    ... )
    >>> print("Coordinates:", train[0])
    Coordinates: (array([3, 1, 0]), array([-1, -3, -4]))
    >>> print("Data:", train[1])
    Data: (array([7, 3, 1]), array([6, 2, 0]))
    >>> print("Weights:", train[2])
    Weights: (array([1, 1, 1]), array([1, 2, 1]))
    >>> print("Coordinates:", test[0])
    Coordinates: (array([2]), array([-2]))
    >>> print("Data:", test[1])
    Data: (array([5]), array([4]))
    >>> print("Weights:", test[2])
    Weights: (array([2]), array([1]))

    Splitting using spatial blocks will use the coordinate information to randomly
    assign data blocks into each group:

    >>> from verde import grid_coordinates
    >>> coordinates = grid_coordinates(region=[0, 3, -4, -1], spacing=1)
    >>> print(coordinates[0])
    [[0. 1. 2. 3.]
     [0. 1. 2. 3.]
     [0. 1. 2. 3.]
     [0. 1. 2. 3.]]
    >>> print(coordinates[1])
    [[-4. -4. -4. -4.]
     [-3. -3. -3. -3.]
     [-2. -2. -2. -2.]
     [-1. -1. -1. -1.]]
    >>> data = np.arange(coordinates[0].size).reshape(coordinates[0].shape)
    >>> print(data)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    >>> # 'spacing' is used to specify the block size
    >>> train, test = train_test_split(
    ...     coordinates, data, method="block", spacing=2, random_state=0
    ... )
    >>> print("Coordinates:", train[0][0], train[0][1], sep="\n  ")
    Coordinates:
      [0. 1. 2. 3. 0. 1. 2. 3. 2. 3. 2. 3.]
      [-4. -4. -4. -4. -3. -3. -3. -3. -2. -2. -1. -1.]
    >>> print("Data:", train[1])
    Data: (array([ 0,  1,  2,  3,  4,  5,  6,  7, 10, 11, 14, 15]),)
    >>> print("Weights:", train[2])
    Weights: (None,)
    >>> print("Coordinates:", test[0][0], test[0][1], sep="\n  ")
    Coordinates:
      [0. 1. 0. 1.]
      [-2. -2. -1. -1.]
    >>> print("Data:", test[1])
    Data: (array([ 8,  9, 12, 13]),)
    >>> print("Weights:", test[2])
    Weights: (None,)

    """
    valid_methods = ["random", "block"]
    if method not in valid_methods:
        raise ValueError(
            "Invalid splitting method '{}'. Must be one of '{}'.".format(
                method, valid_methods
            )
        )
    args = check_fit_input(coordinates, data, weights, unpack=False)
    if method == "random":
        ndata = args[1][0].size
        indices = np.arange(ndata)
        cv = ShuffleSplit(n_splits=1, **kwargs)
        split = next(cv.split(indices))
    if method == "block":
        if "spacing" not in kwargs and "shape" not in kwargs:
            raise ValueError(
                "Method 'block' requires specifying a 'spacing' or 'shape' as well."
            )
        cv = BlockShuffleSplit(n_splits=1, **kwargs)
        split = next(cv.split(coordinates))
    train, test = (tuple(select(i, index) for i in args) for index in split)
    return train, test


def cross_val_score(estimator, coordinates, data, weights=None, cv=None, client=None):
    """
    Score an estimator/gridder using cross-validation.

    Similar to :func:`sklearn.model_selection.cross_val_score` but modified to
    accept spatial multi-component data with weights.

    By default, will use :class:`sklearn.model_selection.KFold` to split the
    dataset. Another cross-validation class can be passed in through the *cv*
    argument.

    Can optionally run in parallel using `dask <https://dask.pydata.org/>`__.
    To do this, pass in a :class:`dask.distributed.Client` as the *client*
    argument. Tasks in this function will be submitted to the dask cluster,
    which can be local. In this case, the resulting scores won't be the actual
    values but :class:`dask.distributed.Future` objects. Call their
    ``.result()`` methods to get back the values or pass them along to other
    dask computations.

    Parameters
    ----------
    estimator : verde gridder
        Any verde gridder class that has the ``fit`` and ``score`` methods.
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...).
    data : array or tuple of arrays
        the data values of each data point. If the data has more than one
        component, *data* must be a tuple of arrays (one for each component).
    weights : none or array or tuple of arrays
        if not none, then the weights assigned to each data point. If more than
        one data component is provided, you must provide a weights array for
        each data component (if not none).
    cv : None or cross-validation generator
        Any scikit-learn or Verde cross-validation generator. Defaults to
        :class:`sklearn.model_selection.KFold`.
    client : None or dask.distributed.Client
        If None, then computations are run serially. Otherwise, should be a
        dask ``Client`` object. It will be used to dispatch computations to the
        dask cluster.

    Returns
    -------
    scores : array
        Array of scores for each split of the cross-validation generator. If
        *client* is not None, then the scores will be futures.

    Examples
    --------

    >>> from verde import grid_coordinates, Trend
    >>> coords = grid_coordinates((0, 10, -10, -5), spacing=0.1)
    >>> data = 10 - coords[0] + 0.5*coords[1]
    >>> # A linear trend should perfectly predict this data
    >>> scores = cross_val_score(Trend(degree=1), coords, data)
    >>> print(', '.join(['{:.2f}'.format(score) for score in scores]))
    1.00, 1.00, 1.00, 1.00, 1.00

    To run parallel, we need to create a :class:`dask.distributed.Client`. It will
    create a local cluster if no arguments are given so we can run the scoring on a
    single machine. We'll use threads instead of processes for this example but in most
    cases you'll want processes.

    >>> from dask.distributed import Client
    >>> client = Client(processes=False)
    >>> # The scoring will now only submit tasks to our local cluster
    >>> scores = cross_val_score(Trend(degree=1), coords, data, client=client)
    >>> # The scores are not the actual values but Futures
    >>> type(scores[0])
    <class 'distributed.client.Future'>
    >>> # We need to call .result() to get back the actual value
    >>> print('{:.2f}'.format(scores[0].result()))
    o
    1.00
    >>> # Close the client and shutdown the local cluster
    >>> client.close()

    """
    coordinates, data, weights = check_fit_input(
        coordinates, data, weights, unpack=False
    )
    if client is None:
        client = DummyClient()
    if cv is None:
        cv = KFold(shuffle=True, random_state=0, n_splits=5)
    if getattr(cv, "is_spatial", False):
        splits = cv.split(coordinates)
    else:
        splits = cv.split(np.arange(data[0].size))
    args = (coordinates, data, weights)
    scores = []
    for train, test in splits:
        train_data, test_data = (
            tuple(select(i, index) for i in args) for index in (train, test)
        )
        score = client.submit(fit_score, estimator, train_data, test_data)
        scores.append(score)
    return np.asarray(scores)


def fit_score(estimator, train_data, test_data):
    """
    Fit an estimator on the training data and then score it on the testing data
    """
    estimator.fit(*train_data)
    return estimator.score(*test_data)


def select(arrays, index):
    """
    Index each array in a tuple of arrays.

    If the arrays tuple contains a ``None``, the entire tuple will be returned
    as is.

    Parameters
    ----------
    arrays : tuple of arrays
    index : array
        An array of indices to select from arrays.

    Returns
    -------
    indexed_arrays : tuple of arrays

    Examples
    --------

    >>> import numpy as np
    >>> select((np.arange(5), np.arange(-3, 2, 1)), [1, 3])
    (array([1, 3]), array([-2,  0]))
    >>> select((None, None, None, None), [1, 2])
    (None, None, None, None)

    """
    if arrays is None or any(i is None for i in arrays):
        return arrays
    return tuple(i.ravel()[index] for i in arrays)

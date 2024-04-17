import numpy as np
import pandas as pd


def geom_mean(x, axis=0):
    """
    calculate the geometric mean of an array

    Parameters
    ----------
    x : ndarray
        1 or 2D array to calculate geometric mean of
    axis : int or str
        the axis along which to take the geometric mean. Accepts
        numpy (0,1) or pandas ('rows', 'columns'). Defaults to 0
        or 'rows'

    Returns
    -------
        geometric mean of x along specified axis.
    """

    if isinstance(x, pd.Series):
        y = x.values.copy()
    elif isinstance(x, pd.DataFrame):
        y = x.values.copy()
    elif isinstance(x, np.ndarray):
        y = x.copy()

    if axis == "rows":
        ax = 0
    elif axis == "columns":
        ax = 1
    else:
        ax = axis

    if len(y.shape) == 1:

        if ax == 0:
            y = y[:, np.newaxis]
            n = y.shape[0]

        elif ax == 1:
            y = y[np.newaxis, :]
            n = y.shape[1]

    else:
        if ax == 0:
            n = y.shape[0]

        elif ax == 1:
            n = y.shape[1]

    g = np.exp(np.sum(np.log(y), axis=ax) / n)

    return g


def ALR(X, reference_column):
    """compute the additive logratio transform from Aitchison (1982)

    Parameters
    ----------
    df : pandas dataframe
        data to be transformed. Should sum to unity.
    reference_column : int or str
        name or location of column to use as the reference
    Returns
    -------
    pandas DataFrame
        ALR transformed pandas dataframe without the reference column data
    """
    if isinstance(X, np.ndarray):
        Y = X.copy()
        ref_idx = reference_column
        out_cols = np.arange(X.shape[1], dtype=int)
        index = np.arange(0, Y.shape[0])

    elif isinstance(X, pd.DataFrame):
        Y = X.values.copy()
        # column number of reference composition for denominator
        ref_idx = np.where(X.columns == reference_column)[0][0]
        out_cols = [col for col in X.columns if not col == reference_column]
        index = X.index

    # dimensions of data
    d = Y.shape[1]

    # divide all columns by the reference denominator column
    alr = np.divide(Y, Y[:, ref_idx][:, np.newaxis])
    # exclude reference column
    alr = alr[:, [i for i in range(d) if not i == ref_idx]]

    return pd.DataFrame(np.log(alr), columns=out_cols, index=index)


def inverse_ALR(alr_df, reference_column):
    """calculate the inverse additive logratio transform
    from Aitchison (1982)

    Parameters
    ----------
    alr_df : pandas DataFrame
        ALR transformed values. output from ALR(function)
    reference_column : str or int
        name or location of column to use as the reference

    Returns
    -------
    pandas DataFrame

    inverse-ALR transformed array, representing original compositions
    between 0 and 1
    """

    df = alr_df.copy()
    df[reference_column] = 0

    df = np.exp(df)

    return df / df.sum(axis=1).values[:, np.newaxis]


def ALR_covariance(X):
    """
    calculate
    1. mean logratio value for each component in a NxD composition
    2. logratio covariance matrix for all components in a NxD composition


    Parameters
    ----------
    X : array-like
        additive logratio transformed values of a NxD composition

    Returns
    -------
    mu : array-like
        mean logratio value for each component
    sigma : array-like
        logratio covariance matrix for all components in NxD composition.

    """
    if isinstance(X, np.ndarray):
        Y = X.copy()
    elif isinstance(X, pd.DataFrame):
        Y = X.values.copy()

    N = Y.shape[0]
    n = N - 1
    IN = np.identity(N)
    JN = np.full(IN.shape, 1)
    jN = JN[:, 0][:, np.newaxis]
    GN = IN - N**-1 * JN

    mu = N**-1 * Y.T @ jN
    # this is from the aitchison textbook and
    # achieves the same as np.cov(X)
    sigma = n**-1 * Y.T @ GN @ Y

    return mu, sigma


def comp_variation_array(X):
    """calculate the compositional variation array
    from Aitchison 1986 Definition 4.3. The upper right triangle
    corresponds to the tau(i,j) symmetrical matrix and the lower
    left triangle corresponds to the xi(i,j) symmetrical matrix
    where:
    tau(ij) = var{log(xi/xj)} (i = 1 ... d; j = 1 ... D)
    xi(ij) = mean{log(xi/xj)} (i = 1 ... d; j = 1 ... D)

    the symmetrical matrix, tau(ij) can also be thought of as
    T, the variation matrix from Definition 4.4

    Parameters
    ----------
    X : array like
        array of compositions NxD

    Returns
    -------
    comp_var_array : array-like

    compositional variation array
    from Aitchison 1986 Definition 4.3. The upper right triangle
    corresponds to the tau(i,j) symmetrical matrix and the lower
    left triangle corresponds to the xi(i,j) symmetrical matrix
    where:
    tau(ij) = var{log(xi/xj)} (i = 1 ... d; j = 1 ... D)
    xi(ij) = mean{log(xi/xj)} (i = 1 ... d; j = 1 ... D)
    """

    if isinstance(X, np.ndarray):
        Y = X.copy()
    elif isinstance(X, pd.DataFrame):
        Y = X.values.copy()

    means = []
    for col in range(Y.shape[1]):
        result = np.log(Y / Y[:, col][:, np.newaxis])
        means.append(result.mean(axis=0))

    xi = np.array(means)
    xi = np.tril(xi)

    variances = []
    for col in range(Y.shape[1]):
        variances.append(np.var(np.log(Y / Y[:, col][:, np.newaxis]), axis=0, ddof=1))

    T = np.array(variances)
    T = np.triu(T)

    return xi + T


def aitchison_distance(comp1, comp2):
    """calculate the aitchison distance between two compositions
    or 1 composition and an array of compositions. Both compositions
    must have the same numbe of components (columns).

    Parameters
    ----------
    comp1 : array like
        1 or 2D array of compositions. Note, if 2D array, comp2
        must be 1D
    comp2 : array like
        1 or 2D array of compositions. Note, if 2D array, comp1
        must be 1D

    Returns
    -------
    array like
        Aitchison distance between input compositions
    """
    if len(comp1.shape) == 2:
        comp2 = comp2[np.newaxis, :]
        assert comp1.shape[1] == comp2.shape[1]

    elif len(comp2.shape) == 2:
        comp1 = comp1[np.newaxis, :]
        assert comp2.shape[1] == comp1.shape[1]

    else:
        assert comp1.shape == comp2.shape

    g1 = geom_mean(comp1, axis="columns")[:, np.newaxis]
    g2 = geom_mean(comp2, axis="columns")[:, np.newaxis]

    d = np.sqrt(np.sum((np.log(comp1 / g1) - np.log(comp2 / g2)) ** 2, axis=1))

    return d


class Composition:
    """

    A class for working with compositional data

    """

    def __init__(self, name):
        self.name = name

    def add_data(self, X, analytes):

        assert isinstance(X, pd.DataFrame)

        self.data = X

        self.analytes = analytes

        self.geometric_mean = pd.Series(
            geom_mean(self.data.loc[:, self.analytes]), index=self.analytes
        )

        self.compositional_variation_array = comp_variation_array(
            self.data.loc[:, analytes]
        )

    def transform_data(self, reference_column, kind="ALR"):
        assert hasattr(self, "data")
        if kind == "ALR":

            self.data_alr = ALR(self.data.loc[:, self.analytes], reference_column)

            self.alr_mu, self.alr_sigma = ALR_covariance(self.data_alr)

            self.alr_mu = pd.Series(self.alr_mu.ravel(), index=self.data_alr.columns)

            self.alr_sigma = pd.DataFrame(
                self.alr_sigma,
                columns=self.data_alr.columns,
                index=self.data_alr.columns,
            )

            self.alr_ref_analyte = reference_column

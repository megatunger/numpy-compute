# Author: DUY ANH
# - Mean, median, standard deviation, min, max
# - Histogram
# - Quantiles (with NaN handling)
# - Axis-wise stats with clear dimension/shape behaviour

import numpy as np

from numcompute.utils import validate_array_like, validate_options, validate_non_empty_array

def handle_nan(arr, handling):
    if handling == 'raise':
        if np.isnan(arr).any():
            raise ValueError("There are NaN values in inputs")
    elif handling =='return_nan':
        return arr
    elif type(handling) in [int, float, bool, np.ndarray]:
        # Impute given data
        arr[np.isnan(arr)] = handling
        return arr
    else:
        # Raise error if encountered unexpected value
        raise ValueError(f"Expect nan_handling to be 'raise', or [int, float, bool, np.ndarray] for imputation. Got {type(handling)}")

    return arr
    
def mean(arr, nan_handling="raise", *args, **kwargs) -> np.ndarray:
    """Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    This function is equivalent to numpy.mean with additional NaN handling

    Parameters
    ----------
    arr : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    nan_handling : str or int or float or bool
        How to handle NaN values. If 'ignore', NaN values are ignored and
        output only computed on remaining values. Otherwise, the array 
        will be imputed with the given value.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.
    """
    arr = validate_array_like(arr)
    arr = validate_non_empty_array(arr)
    arr = handle_nan(arr, nan_handling)

    return np.mean(arr, *args, **kwargs)


def median(arr, nan_handling="raise", *args, **kwargs) -> np.ndarray:
    """Compute the median along the specified axis.

    Returns the median of the array elements. 
    
    This function is equivalent to numpy.median with additional NaN handling

    Parameters
    ----------
    arr : array_like
        Input array or object that can be converted to an array.
    nan_handling : str or int or float or bool
        How to handle NaN values. If 'ignore', NaN values are ignored and
        output only computed on remaining values. Otherwise, the array 
        will be imputed with the given value.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default,
        axis=None, will compute the median along a flattened version of
        the array. If a sequence of axes, the array is first flattened
        along the given axes, then the median is computed along the
        resulting flattened axis.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.
    """
    arr = validate_array_like(arr)
    arr = validate_non_empty_array(arr)
    arr = handle_nan(arr, nan_handling)
    
    return np.median(arr, *args, **kwargs)


def std(arr, nan_handling="raise", *args, **kwargs) -> np.ndarray:
    """Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.
    
    This function is equivalent to numpy.std with additional NaN handling
    

    Parameters
    ----------
    arr : array_like
        Calculate the standard deviation of these values.
    nan_handling : str or int or float or bool
        How to handle NaN values. If 'ignore', NaN values are ignored and
        output only computed on remaining values. Otherwise, the array 
        will be imputed with the given value.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero. See Notes for details about use of `ddof`.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the standard deviation.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this std function.

        .. versionadded:: 2.0.0

    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.
    """
    arr = validate_array_like(arr)
    arr = validate_non_empty_array(arr)
    arr = handle_nan(arr, nan_handling)

    return np.std(arr, *args, **kwargs)


def min(arr, nan_handling="raise", *args, **kwargs) -> np.ndarray:
    """Return the minimum of an array or minimum along an axis.

    This function is equivalent to numpy.min with additional NaN handling

    Parameters
    ----------
    arr : array_like
        Input data.
    nan_handling : str or int or float or bool
        How to handle NaN values. If 'ignore', NaN values are ignored and
        output only computed on remaining values. Otherwise, the array 
        will be imputed with the given value.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``min`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    min : ndarray or scalar
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``.  If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.
    """
    arr = validate_array_like(arr)
    arr = validate_non_empty_array(arr)
    arr = handle_nan(arr, nan_handling)

    return np.min(arr, *args, **kwargs)


def max(arr, nan_handling="raise", *args, **kwargs) -> np.ndarray:
    """Return the maximum of an array or maximum along an axis.

    This function is equivalent to numpy.max with additional NaN handling

    Parameters
    ----------
    arr : array_like
        Input data.
    nan_handling : str or int or float or bool
        How to handle NaN values. If 'ignore', NaN values are ignored and
        output only computed on remaining values. Otherwise, the array 
        will be imputed with the given value.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used. If this is a tuple of ints, the maximum is selected over
        multiple axes, instead of a single axis or all the axes as before.

    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``max`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    max : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.
    """
    arr = validate_array_like(arr)
    arr = validate_non_empty_array(arr)
    arr = handle_nan(arr, nan_handling)

    return np.max(arr, *args, **kwargs)


def histogram(arr, nan_handling="raise", *args, **kwargs) -> np.ndarray:
    """Compute the histogram of a dataset.

    This function is equivalent to numpy.histogram with additional NaN handling
        
    Parameters
    ----------
    arr : array_like
        Input data. The histogram is computed over the flattened array.
    nan_handling : str or int or float or bool
        How to handle NaN values. If 'ignore', NaN values are ignored and
        output only computed on remaining values. Otherwise, the array 
        will be imputed with the given value.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
        Please note that the ``dtype`` of `weights` will also become the
        ``dtype`` of the returned accumulator (`hist`), so it must be
        large enough to hold accumulated values as well.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.

    Returns
    -------
    hist : array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.  If `weights` are given,
        ``hist.dtype`` will be taken from `weights`.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    """
    arr = validate_array_like(arr)
    arr = validate_non_empty_array(arr)
    arr = handle_nan(arr, nan_handling)

    return np.histogram(arr, *args, **kwargs)

def quantile(arr, q, nan_handling="raise", *args, **kwargs) -> np.ndarray:
    """Compute the q-th quantile of the data along the specified axis.
    
    This function is equivalent to numpy.quantile with additional NaN handling

    Parameters
    ----------
    arr : array_like of real numbers
        Input array or object that can be converted to an array.
    nan_handling : str or int or float or bool
        How to handle NaN values. If 'ignore', NaN values are ignored and
        output only computed on remaining values. Otherwise, the array 
        will be imputed with the given value.
    q : array_like of float
        Probability or sequence of probabilities of the quantiles to compute.
        Values must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is
        to compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and buffer length as the expected output, but the
        type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  There are many different methods, some unique to NumPy.
        The recommended options, numbered as they appear in [1]_, are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous. For backward compatibility
        with previous versions of NumPy, the following discontinuous variations
        of the default 'linear' (7.) option are available:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        See Notes for details.

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the quantile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.
        See the notes for more details.

        .. versionadded:: 2.0.0

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single probability and `axis=None`, then the result
        is a scalar. If multiple probability levels are given, first axis
        of the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.
    """
    arr = validate_array_like(arr)
    arr = validate_non_empty_array(arr)
    arr = handle_nan(arr, nan_handling)
    return np.quantile(arr, q, *args, **kwargs)

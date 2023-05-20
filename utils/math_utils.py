import torch


def z_score(x, mean, std):
    """
    Z-score normalization function: $Z = (X - \mu) / \sigma $
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: torch array, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: torch array, the z-score normalized array.
    """
    return (x - mean) / std


def un_z_score(x_normed, mean, std):
    """
    Undo the z-score calculation
    :param x_normed torch array, input_array to be un-normalized
    :param mean: float, the value of mean
    :param std: float, the value of standard deviation
    """
    return x_normed * std + mean


def RMSE(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth
    :param v_: torch array, prediction
    :return: torch scaler, RMSE averages on all elements of input
    """
    return torch.sqrt(torch.mean(v_ - v) ** 2)


def MAE(v, v_):
    """
    Mean Absolute Error
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v))


def MAPE(v, v_):
    """
    Mean absolute percentage error, given as a % (e.g. 99 -> 99%)
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAPE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v) / (v + 1e-15) * 100)

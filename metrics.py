from sklearn.metrics import log_loss
from helperFunctions.sampling import downsample_recal

def normalized_log_loss(y_true, y_pred, background_rate, **kwargs):
    model_loss = log_loss(y_true, y_pred, **kwargs)
    background_loss = log_loss(y_true, [background_rate]*len(y_true), **kwargs)
    return model_loss/background_loss


def normalized_log_loss_for_scorer(y_true, y_pred, **kwargs):

    """
    Version of the normalized log loss function that can be used in sklearn's make_scorer.
    It needed to take only two arguments other than **kwargs. Downsample recalibration must be done
    inside this function instead of before hand for it to work inside sklearn's CV.
    :param y_true: True labels of test data
    :param y_pred: Predicted labels of test data
    :param kwargs:
        background_rate: rate of positive examples in data set for normalization
        recalibrate: (boolean) weather or not to recalibrate predicted
                probabilities due to downsampling
        downsamp_rate: the rate of downsampling to be used in probability recalibration.
            if not present, the background_rate will be used.
    :return: float logistic loss of model normalized to the logistic loss of the background
        positive probability.
    """
    try:
        background_rate = kwargs['background_rate']
        background_loss = log_loss(y_true, [background_rate] * len(y_true))
    except:
        print('background rate not given in kwargs, returning standard log loss.')
        background_loss = 1

    try:
        downsamp_rate = kwargs['downsamp_rate']
    except:
        downsamp_rate = background_rate

    try:
        if kwargs['recalibrate']:
            try:
                y_pred = downsample_recal(y_pred, downsamp_rate)
            except:
                print('noo')
    except:
        pass

    print(y_pred)
    model_loss = log_loss(y_true, y_pred)
    return model_loss/background_loss


import helperFunctions.sampling as samp
import numpy as np


def predict_downsampled_probas(model, X, ds_rate):
    p = model.predict_proba(X)[:, 1]
    return samp.downsample_recal(p, ds_rate)


def predict_from_models(models, X, ds_rate):

    if not isinstance(models, (list)):
        models = [models]

    mdl_probas = []
    for mdl in models:
        predict_downsampled_probas(mdl, X, ds_rate)
        mdl_probas.append(mdl)

    proba_mat = np.matrix(mdl_probas)
    mean_probas = proba_mat.mean(0).tolist()[0]
    return mean_probas


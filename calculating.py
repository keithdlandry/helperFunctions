import numpy as np
import pandas as pd
# from scipy.special import comb
from helperFunctions.miscellaneous import list_to_comma_sep



def previous_rolling_quant(df, x, y, window, shift=1, quantile=50, fill_with_expanding=False):

    colname = x + '_{}roll_quant{}'.format(window, quantile)
    df[colname] = df.groupby(x)[y] \
        .apply(lambda x: x.shift(shift).rolling(window).apply(np.percentile, args=(quantile,)))

    if fill_with_expanding:
        expand = df.groupby(x)[y] \
            .apply(lambda x: x.shift(shift).expanding().apply(np.nanpercentile, args=(quantile,)))
        df[colname] = np.where(pd.isnull(df[colname]), expand, df[colname])


def previous_rolling_mean(df, x, y, window, shift=1, fill_with_expanding=False):

    if not isinstance(x, str):
        name = list_to_comma_sep(x)
    else:
        name = x

    colname = name + '_{}roll_mean'.format(window)
    df[colname] = df.groupby(x)[y] \
        .apply(lambda x: x.shift(shift).rolling(window).mean())

    if fill_with_expanding:
        expand = df.groupby(x)[y] \
            .apply(lambda x: x.shift(shift).expanding().mean())
        df[colname] = np.where(pd.isnull(df[colname]), expand, df[colname])

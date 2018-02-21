import pandas as pd
from helperFunctions.manipulating import append_to_column_name
from helperFunctions.miscellaneous import make_list_if_not_list
import numpy as np


def find_frequent_vals(data, n, type='top'):

    # type can be 'top', 'usage', 'percent'
    if type == 'top':
        freq_vals = data.value_counts()[:n]
    elif type == 'usage':
        freq_vals = data.value_counts()[data.value_counts() > n]
    else:
        freq_vals = data.value_counts()[data.value_counts() > n*len(data)]

    return freq_vals


def to_categorical(df, columns): # maybe don't even need a function
    df[columns] = df[columns].apply(lambda x: x.astype('category'))
    return df


def to_binary(n, length=0):
    s = '0:0{}b'.format(length)
    s = '{' + s + '}'
    return s.format(n)


def to_integer_label(column):
    values = column.unique()
    # convert to integers via dictionary replacement
    d = {v:i for i, v in enumerate(values)}
    # fasted way to replace col vales with dict
    # requires all column values be in the dictionary
    return column.map(d.get)


def encode_column_binary(column, prefix):

    column = to_integer_label(column)
    # convert ints to binary with leading zeros
    max_val = column.max()
    max_len = len(to_binary(max_val))
    binary_col = [list(to_binary(n, max_len)) for n in column]

    new_col_names = ['{}_b{}'.format(prefix, i) for i in range(max_len)]
    binary_df = pd.DataFrame(binary_col, columns=new_col_names)
    return binary_df


def encode_prev_rate(df, columns, dependent_var='y', nmin=None):
    """
    Get the previous rate of positive examples for a feature or combination of features
    up until the row in question.
    :param df:
    :param columns:
    :param dependent_var:
    :return:
    """
    df = df.copy(deep=True)

    if not isinstance(columns, (list)):
        cname = columns + '_prev_rate'
    else:
        cname = '_'.join(columns) + '_prev_rate'

    df['pos'] = df.groupby(columns)[dependent_var].cumsum() \
        .shift(1).fillna(0) \
        .astype(df[dependent_var].dtype)
    df['-1s'] = 1
    df['tot'] = df.groupby(columns)['-1s'].cumsum() \
        .shift(1).fillna(0) \
        .astype(df[dependent_var].dtype)
    df[cname] = df['pos']/df['tot']
    if nmin is not None:
        df.loc[df.tot <= nmin, cname] = 0
    return df[[cname]]


def encode_column(df, column, method='one_hot', n=None,
                  frequency_type=None, dependent_var=None, dummy_nan=False):
    """
    encodes a column of a dataframe
    :param df: df to be encoded
    :param column: name of column to be encoded
    :param method: encoding method - 'one_hot', 'binary', 'diff_mean_dep'
    :param n: either number of most frequent levels to be encoded (top),
        number of rows a level must have to be encoded (usage),
        or the percentage of rows a level must have to be encoded (percentage)
        depending on frequency_type
    :param frequency_type: either 'top', 'usage', or 'percentage'
    :param dependent_var: the dependent variable is required only for
        difference in mean dependent variable and previous rate encoding
    :return: Pandas DataFrame of encoded column
    """
    df = df.copy(deep=True)
    if frequency_type is not None and n is not None:
        if frequency_type == 'top':
            print('encoding {} most frequent levels'.format(n))
        elif frequency_type == 'usage':
            print('encoding all levels with more than {} examples'.format(n))
        elif frequency_type == 'percentage':
            print('encoding all levels which appear in at least {} percent of rows'.format(n))
        else:
            raise ValueError('invalid frequency type: must be one of "top", "usage", "percent"')

        freq_vals = find_frequent_vals(df[column], n, frequency_type).index
        othername = 'aaall_other_levels'
        # pdb.set_trace()
        if df[column].dtype.name == 'category':
            df[column] = df[column].cat.add_categories([othername])
        df.loc[~df[column].isin(freq_vals), column] = othername

    if method == 'one_hot':
        if dummy_nan:
            encoded_df = encode_one_hot_nans(df, column)
        else:
            encoded_df = (pd.get_dummies(df[column], prefix=column))
    elif method == 'binary':
        encoded_df = encode_column_binary(df[column], prefix=column)
    elif method == 'diff_mean_dep':
        if dependent_var is None:
            raise ValueError(
                'dependent_var must be set for difference in mean of dependent value encoding')
        encoded_df = encode_diff_mean_dep(df, column, dependent_var)
    elif method == 'prev_rate':
        encoded_df = encode_prev_rate(df, column, dependent_var)
    else:
        raise ValueError(
            'invalid encoding method: must be one of "one_hot", "binary", "diff_mean_dep"')
    return encoded_df


def encode_one_hot_nans(df, column):
    # Slower than pd.get_dummies but allows for [1, NaN] encoding
    # instead of [1, 0] encoding. Faster than using pd.get_dummies followed
    # by changing the zeros to np.nan
    df = df.copy(deep=True)
    c = '___oNe___'
    df[c] = 1
    encoded_df = df[[column, c]].pivot(columns=column, values=c)
    encoded_df.columns = [str(cname) for cname in encoded_df.columns]
    append_to_column_name(encoded_df, encoded_df.columns, '{}_'.format(column))
    return encoded_df


def encode_from_old(df, column, old_vals, method='one_hot'):

    df = df.copy(deep=True)

    othername = 'aaall_other_levels'
    if df[column].dtype.name == 'category':
        df[column] = df[column].cat.add_categories([othername])
    df.loc[~df[column].isin(old_vals), column] = othername

    if method == 'one_hot':
        encoded_df = (pd.get_dummies(df[column], prefix=column))
    elif method == 'binary':
        encoded_df = encode_column_binary(df[column], prefix=column)
    elif method == 'diff_mean_dep':
        if dependent_var is None:
            raise ValueError(
                'dependent_var must be set for difference in mean of dependent value encoding')
        encoded_df = encode_diff_mean_dep(df, column, dependent_var)
    elif method == 'prev_rate':
        encoded_df = encode_prev_rate(df, column, dependent_var)
    else:
        raise ValueError(
            'invalid encoding method: must be one of "one_hot", "binary", "diff_mean_dep"')
    return encoded_df


def encode_datetime(df, datetime_col_name, features=['hour', 'day'], suffix='',
                    make_categorical=False, drop_datetime=True, inplace=False):

    features = make_list_if_not_list(features)

    if not isinstance(df[datetime_col_name].iloc[0], (pd._libs.tslib.Timestamp)):
        print('Converting to pandas._libs.tslib.Timestamp')
        df[datetime_col_name] = pd.to_datetime(df[datetime_col_name])

    if not inplace:
        df = df.copy(deep=True)

    # pandas dt.dayofweek monday = 0 sunday = 6 for some reason
    column_names = [f + suffix for f in features]
    feature_dict = dict(zip(features, column_names))
    datecol = df[datetime_col_name]

    for key in feature_dict.keys():
        df[feature_dict[key]] = datecol.dt.__getattribute__(key)

    if drop_datetime:
        df.drop(datetime_col_name, axis=1, inplace=True)
    if make_categorical:
        df = to_categorical(df, features)

    if not inplace:
        return df


def make_equiv_columns(df, col_names, fill=np.nan, inplace=False):

    if not inplace:
        df = df.copy(deep=True)
    current_cols = set(df.columns)
    all_cols = set(col_names)
    cols_needed = all_cols.difference(current_cols)
    if len(cols_needed) > 0:
        print('Found {} missing columns.'.format(len(cols_needed)))
        print('Adding columns:', cols_needed)
        print('to ensure same columns as data set used for model training.')
        for c in cols_needed:
            df[c] = fill

    extra_cols = current_cols.difference(all_cols)
    if len(extra_cols) > 0:
        print('Found {} extra columns.'.format(len(extra_cols)))
        print('Removing columns:', extra_cols)
        print('to ensure same columns as data set used for model training.')
        df.drop(extra_cols, axis=1, inplace=True)

    if not inplace:
        return df[col_names]


def encode_column_angular(column):

    pi = np.pi

    min_val = np.min(column)
    max_val = np.max(column)

    adjusted_max = max_val - min_val

    sin = np.sin(column*2*pi/adjusted_max)
    cos = np.cos(column*2*pi/adjusted_max)

    return [find_angle(s, c) for s, c in zip(sin, cos)]


def encode_trigonometric(column, range=None):

    pi = np.pi

    if range is None:
        min_val = np.min(column)
        max_val = np.max(column)
    else:
        min_val = range[0]
        max_val = range[1]

    adjusted_max = max_val - min_val
    sin = np.sin(column * 2 * pi / adjusted_max)
    cos = np.cos(column * 2 * pi / adjusted_max)

    return sin, cos


def encode_cyclic_linear(column, range=None):

    if range is None:
        min_val = np.min(column)
        max_val = np.max(column)
    else:
        min_val = range[0]
        max_val = range[1]

    adjusted_max = max_val - min_val

    return find_yval1(column, adjusted_max), find_yval2(column, adjusted_max)


def find_yval1(X, alpha):

    Y = []
    for x in X:

        if x <= alpha / 4:
            y = x * 4 / alpha
        elif x <= 3 * alpha / 4:
            y = -x * 4 / alpha + 2
        else:
            y = x * 4 / alpha - 4

        Y.append(y)
    return Y


def find_yval2(X, alpha):

    Y = []
    for x in X:

        if x <= alpha / 2:
            y = -x * 4 / alpha + 1
        else:
            y = x * 4 / alpha - 3

        Y.append(y)
    return Y


def find_angle(sin, cos):

    if sin >= 0:
        theta = np.arccos(cos)
    elif sin < 0:
        theta = 2*np.pi - np.arccos(cos)

    return theta


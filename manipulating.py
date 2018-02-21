import numpy as np
import pandas as pd
from helperFunctions.miscellaneous import make_list_if_not_list
from helperFunctions.miscellaneous import strip_suffix


def flatten_hier_column_names(df, delim='_'):

    df = df.copy()

    # make sure all columns are a string.
    # not the case generally if grouping on a column containing a number
    col_strs = [[str(c) for c in hier_col_name] for hier_col_name in df.columns.values]
    df.columns = [strip_suffix(delim.join(c).strip(), delim) for c in col_strs]

    return df


def convert_to_datetime(df, columns):
    df = df.copy(deep=True)
    df[columns] = df[columns].apply(lambda col: pd.to_datetime(col, unit='ms'), axis=1)
    return df


def drop_cols_between(df, first_col, last_col):

    df = df.copy()
    rem_cols = df.loc[:, first_col:last_col].columns.tolist()
    df.drop(rem_cols, inplace=True, axis=1)
    return df


def subset_by_indx(data, indices):
    # Function to subset either a DataFrame or Array like by index.

    try:
        sub = data.iloc[indices]
    except AttributeError:
        sub = data[indices]
    finally:
        return sub


def remove_training_cols(train=None, test=None, training_tag='_train'):
    # This is a function that allows for training and testing to utilize different
    # columns in the DataFrame. The DataFrame will start with a column marked with a
    # training tag (e.g. _train) denoting this column should replace the column with
    # the same name but without the tag for training and be removed for testing.
    # This way the feature names will be the same for training and test sets.
    # See example below:
    # Y   | x1 | x2  | x2_train
    # 1.5 | 2  | 4.2 | 5
    # 2.8 | 3  | 5.1 | 5.7
    # if it is the testing set will become:
    # Y   | x1 | x2
    # 1.5 | 2  | 4.2
    # 2.8 | 3  | 5.1
    # or if it is the training set:
    # Y   | x1 | x2
    # 1.5 | 2  | 5
    # 2.8 | 3  | 5.7

    # Remove columns with suffix in test set
    if test is not None:
        training_mean_cols = test.columns[test.columns.str.contains(training_tag)]
        test.drop(training_mean_cols, axis=1, inplace=True)

    # Columns are overwritten by the training values and the training columns are removed
    if train is not None:
        train = remove_suffix_from_col_names(train, training_tag, drop_cols=True)

    return train, test


def replace_string_in_col_name(df, original, new, drop_cols=False):

    df = df.copy()
    columns_with_str = df.columns[df.columns.str.contains(original)]
    columns = [c.replace(original, new) for c in columns_with_str]

    for col, col_str in zip(columns, columns_with_str):
        df[col] = df[col_str]

    if drop_cols:
        df.drop(columns_with_str, axis=1, inplace=True)
    return df


def remove_suffix_from_col_names(df, suffix, drop_cols=False):
    # TODO: in rare case where suffix is also contained in the column name (e.g. train_size_train)
    # TODO: the replace will replace both occurances. Make this more robust.
    # TODO: test this fix
    columns_suffix = df.columns[df.columns.str.contains(suffix)]
    # columns = [c.replace(suffix, '') for c in columns_suffix]
    columns = [strip_suffix(c, suffix) for c in columns_suffix]


    for col, col_suf in zip(columns, columns_suffix):
        df[col] = df[col_suf]

    if drop_cols:
        df.drop(columns_suffix, axis=1, inplace=True)
    return df


def group_agg_pivot_df(df, group_cols, agg_func='count', agg_col=None, fillna=True):

    if agg_col is None:
        agg_col = group_cols[0]

    grouped = df.groupby(group_cols).agg({agg_col: agg_func})
    # levels to unstack. Don't include the first level since
    # we want the to keep identifying index unstack levels : [1, 2, .... , n]
    unstack_lvls = list(range(len(group_cols)))[1:]
    grouped = grouped.unstack(level=unstack_lvls)
    if fillna:
        grouped = grouped.fillna(0)
    # drop aggregation column name from hierarchical column names
    grouped.columns = grouped.columns.droplevel()

    # If more than two grouped columns, hierarchical columns are created.
    # Group these columns names into a single string, remove hierarchical
    # columns in favor of single new column.
    if len(group_cols) > 2:
        # make sure all columns are a string.
        # not the case generally if you group on a column containing a number
        col_strs = [[str(c) for c in hier_col_name] for hier_col_name in grouped.columns.values]
        # join the hierarchical columns to a single string and remove leading/trailing whitespace
        grouped.columns = ['_'.join(c).strip() for c in col_strs]

    # promote index to column (the first element of group_cols)
    pivot_df = grouped.reset_index()
    # remove spaces in column names in favor of underscore
    # need to convert to string here only if grouping on two columns and second is integer
    pivot_df.columns = [str(s).replace(' ', '_').lower() for s in pivot_df.columns]
    return pivot_df


def split_stack_df(df, id_cols, split_col, new_col_name):
    # id_cols are the columns we want to pair with the values
    # from the split column

    # example   id | split_col
    #           12 | a,b
    #
    #           id | new_col
    #           12 | a
    #           12 | b

    stacked = df.set_index(id_cols)[split_col].str.split(',', expand=True) \
        .stack().reset_index(level=id_cols)
    stacked.columns = id_cols + [new_col_name]
    return stacked


def append_to_column_name(df, cols, string_to_append, before=True):

    cols = make_list_if_not_list(cols)

    if all(isinstance(i, str) for i in cols):
        old_names = df.columns[df.columns.isin(cols)]
    elif all(isinstance(i, int) for i in cols):
        old_names = df.columns[cols]
    else:
        raise ValueError('cols should either be a list of indices or a list of column names')
    if before:
        new_names = [string_to_append + o for o in old_names]
    else:
        new_names = [o + string_to_append for o in old_names]

    rename_dict = dict(zip(old_names, new_names))
    df.rename(columns=rename_dict, inplace=True)


def assign_groups(df, col_to_bin, method, sig_delta=.5, sig_limits=[-5, 5], bins=10, centered=False):
    df = df.copy(deep=True)

    if method == 'sigma':

        sig = df[col_to_bin].std()
        mu = df[col_to_bin].mean()

        for i, x in enumerate(np.arange(sig_limits[0], sig_limits[1], sig_delta)):
            # print(x, x + sig_delta,  mu + x*sig, mu + x*sig + sig_delta*sig,
            #       mu + x * sig -sig_delta*sig/2, mu + x * sig + sig_delta * sig/2)
            if centered:
                df.loc[
                    (df[col_to_bin] > mu + x * sig - sig_delta * sig / 2) &
                    (df[col_to_bin] < mu + x * sig + sig_delta * sig / 2), 'group'] = i
            else:
                df.loc[
                    (df[col_to_bin] > mu + x * sig) &
                    (df[col_to_bin] < mu + x * sig + sig_delta * sig), 'group'] = i

    elif method == 'fixed_width' or method == 'user_defined':
        if isinstance(bins, int):
            labs = range(bins)
        else:
            try:
                labs = range(len(bins))
            except:
                raise ValueError('bins must be an integer or iterable')
        df['group'] = pd.cut(df[col_to_bin], bins, labels=labs)

    elif method == 'constant':
        pass

    return df


# testing to make agg pivot function faster
# i forget if it worked
def f2(df, index, columns):
    df['one'] = 1
    piv = pd.pivot_table(df, index=index, columns=columns,
                   aggfunc=np.sum)
    col_strs = [[str(c) for c in hier_col_name] for hier_col_name in piv.columns.values]
    # join the hierarchical columns to a single string and remove leading/trailing whitespace
    piv.columns = ['_'.join(c).strip() for c in col_strs]
    piv.columns = [str(s).replace(' ', '_').lower() for s in piv.columns]
    piv = piv.reset_index().fillna(0)
    del df['one']
    return piv





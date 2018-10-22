import numpy as np
import pandas as pd
from helperFunctions.miscellaneous import make_list_if_not_list
from helperFunctions.miscellaneous import strip_suffix

"""
Functions that help with pandas DataFrames. Taking over from manipulating.
"""


def read_df(file, remove_unnamed=True, **kwargs):

    df = pd.read_csv(file, **kwargs)
    if remove_unnamed:
        drops = [col for col in df.columns if 'Unnamed: ' in col]
        df = df.drop(drops, axis=1)

    return df


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


def flatten_hier_column_names(df, delim='_'):

    # if isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
    if isinstance(df.columns, pd.indexes.multi.MultiIndex):
        df = df.copy()

        # make sure all columns are a string.
        # not the case generally if grouping on a column containing a number
        col_strs = [[str(c) for c in hier_col_name] for hier_col_name in df.columns.values]
        df.columns = [strip_suffix(delim.join(c).strip(), delim) for c in col_strs]

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
    # TODO: think it's fixed but needs more testing
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


def unmelt(df, values, columns, index=None, reset_index=True, remove_multiindex=True):

    """
    This function unmelts a dataframe

    :param df:
    :param values:
    :param columns:
    :param index:
    :param reset_index:
    :return:
    """

    # if no index is provided use all other columns in DataFrame
    if index is None:
        index = set(df.columns)
        _cols = make_list_if_not_list(columns)
        _vals = make_list_if_not_list(values)
        index = index - set(_cols + _vals)
    # pivot table to unmelt the dataframe using first item found for each group
    dfpiv = df.pivot_table(values=values, columns=columns, index=index, aggfunc='first')
    if reset_index:
        dfpiv.reset_index(inplace=True)
    if remove_multiindex:
        dfpiv = flatten_hier_column_names(dfpiv)
    # remove column index name in case one was created
    dfpiv.columns.name = ''
    return dfpiv

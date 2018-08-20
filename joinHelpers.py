import warnings
import numpy as np
from helperFunctions.miscellaneous import make_list_if_not_list

'''
Function which help with joins using 
Pandas DataFrames
'''

# TODO: pandas merge is sometimes incosistent with different dtypes (or even sometimes with the same dtype if "mixed".
# TODO: Build checks to warn if this is the case.
# TODO: new function for column selection and printing (with verbosity options)

def check_types(df1, df2, cols1, cols2):

    same = df1[cols1].dtypes == df2[cols2].dtypes
    return same


def left_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None, drop_right_on=False, verbosity=1, **kwargs):
    # join by common columns if nothing is specified
    if on == left_on == right_on is None:
        if verbosity == 1:
            print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        if verbosity == 1:
            print('left joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on is not None:
        left_on = right_on = on

    left_on = make_list_if_not_list(left_on)
    right_on = make_list_if_not_list(right_on)

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')

    dataframe1 = dataframe1.merge(dataframe2, left_on=left_on, right_on=right_on, how='left', **kwargs)

    if drop_right_on:
        if on is not None:
            warnings.warn('Can not drop right joining columns if they are all in both DataFrames')
        else:
            drops = set(right_on).difference(left_on)
            dataframe1.drop(drops, axis=1, inplace=True)

    return dataframe1


''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''


def inner_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None):
    # join by common columns if nothing is specified
    if on == left_on == right_on is None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('inner joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on is not None:
        left_on = right_on = on

    left_on = make_list_if_not_list(left_on)
    right_on = make_list_if_not_list(right_on)

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')

    return dataframe1.merge(dataframe2, left_on=left_on, right_on=right_on, how='inner')

''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''


def outer_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None):
    # join by common columns if nothing is specified
    if on == left_on == right_on is None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('outer joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on is not None:
        left_on = right_on = on

    left_on = make_list_if_not_list(left_on)
    right_on = make_list_if_not_list(right_on)

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')

    return dataframe1.merge(dataframe2, left_on=left_on, right_on=right_on, how='outer')

''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''

def semi_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None):

    # TODO: seems like a simple .isin() is faster? Test this out and change if neccessary.
    # join by common columns if nothing is specified
    if on == left_on == right_on is None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('semi joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on is not None:
        left_on = right_on = on

    left_on = make_list_if_not_list(left_on)
    right_on = make_list_if_not_list(right_on)

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')


    dataframe2['_jointag_'] = 'tag'                          # add a tag in case of joining on all columns (if all cols no NA to drop)
    df2_nodups     = dataframe2.drop_duplicates(right_on)    # drop duplicates so as not to add any rows during merge
    df2_nodups = df2_nodups[right_on + ['_jointag_']]        # subset just the columns of interest in case df is very wide
    # merge resets index
    merged = dataframe1.merge(df2_nodups, left_on=left_on,
                              right_on=right_on, how='inner')
    if len(merged) == 0:
        print('Warning: semi join is returning and empty dataframe')
    return merged[dataframe1.columns]

''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''


def anti_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None, reset_index=True):
    # join by common columns if nothing is specified
    if on == left_on == right_on is None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('anti joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on is not None:
        left_on = right_on = on

    left_on = make_list_if_not_list(left_on)
    right_on = make_list_if_not_list(right_on)

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')

    # pdb.set_trace()
    dataframe2['_jointag_'] = 'tag'                          # add a tag in case of joining on all columns (if all cols no NA to drop)
    df2_nodups     = dataframe2.drop_duplicates(right_on)    # drop duplicates so as not to add any rows during merge
    df2_nodups = df2_nodups[right_on + ['_jointag_']]        # subset just the columns of interest in case df is very wide
    dataframe1.reset_index(drop=True, inplace=True)
    all_df1_ids    = list(dataframe1.index)
    df1_ids_in_df2 = list(dataframe1.merge(df2_nodups,
                                           left_on=left_on,
                                           right_on=right_on,
                                           how='left').dropna(subset=['_jointag_']).index)
    ids_not_in_df2 = set(all_df1_ids).difference(df1_ids_in_df2)

    dataframe2.drop('_jointag_', axis=1, inplace=True)

    if len(ids_not_in_df2) == 0:
        print('Warning: anti join is returning and empty dataframe')
    if reset_index:
        return dataframe1.iloc[list(ids_not_in_df2)].reset_index(drop=True)
    else:
        return dataframe1.iloc[list(ids_not_in_df2)]

''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''

def bad_semi_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None):
    # join by common columns if nothing is specified
    if on == left_on == right_on is None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('semi joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on is not None:
        left_on = right_on = on

    left_on = make_list_if_not_list(left_on)
    right_on = make_list_if_not_list(right_on)

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')

    dataframe2['_jointag_'] = 'tag'                    # add a tag in case of joining on all columns (if all cols no NA to drop)
    df2_nodups = dataframe2.drop_duplicates(right_on)  # drop duplicates so as not to add any rows during merge
    df2_nodups = df2_nodups[right_on + ['_jointag_']]  # subset just the columns of interest in case df is very wide
    # merge resets index
    df1_ids_in_df2 = list(dataframe1.merge(df2_nodups,
                                           left_on=left_on,
                                           right_on=right_on,
                                           how='left').dropna(subset=['_jointag_']).index)

    dataframe2.drop('_jointag_', axis=1, inplace=True)

    if len(df1_ids_in_df2) == 0:
        print('Warning: anti join is returning and empty dataframe')

    return dataframe1.loc[list(df1_ids_in_df2)].reset_index(drop=True)

''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''

def left_interlace(dataframe1, dataframe2, interlace_cols, on=None, left_on=None, right_on=None,
                   drop_right_on=False, **kwargs):

    interlace_cols = make_list_if_not_list(interlace_cols)

    if on == left_on == right_on is None:
        print('No columns specified')
        d1cols = dataframe1.columns
        d2cols = dataframe2.columns
        d2cols = [c for c in d2cols if c not in interlace_cols]
        common_cols = list(set(d2cols).intersection(d1cols))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('left joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on is not None:
        left_on = right_on = on

    if any([lo in interlace_cols for lo in left_on]):
        raise ValueError('interlace columns found in left joining columns')
    if any([ro in interlace_cols for ro in right_on]):
        raise ValueError('interlace columns found in right joining columns')

    left_on = make_list_if_not_list(left_on)
    right_on = make_list_if_not_list(right_on)

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')

    dataframe1 = dataframe1.merge(dataframe2, left_on=left_on,
                                  right_on=right_on, how='left', **kwargs)

    for col in interlace_cols:
        dataframe1[col] = np.where(dataframe1[col+'_x'].notnull(),
                                   dataframe1[col+'_x'], dataframe1[col+'_y'])
        dataframe1.drop([col+'_x', col+'_y'], axis=1, inplace=True)

    return dataframe1

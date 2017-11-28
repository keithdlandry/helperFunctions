import pdb
'''
Function which help with joins using 
Pandas DataFrames
'''
# TODO: left_on, right_on, on should also take str as well as list or at least throw an error when str is passed
# TODO: this is done for left join. just copy lines 22 to 25


def left_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None):
    # join by common columns if nothing is specified
    if on == left_on == right_on == None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('left joining on: ', common_cols)
        on = common_cols
        
    # check if on was specified and create left_on/right_on
    if on != None:
        left_on = right_on = on

    if isinstance(left_on, (str)):
        left_on = [left_on]
    if isinstance(right_on, (str)):
        right_on = [right_on]
        
    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')
        
    return dataframe1.merge(dataframe2, left_on=left_on, right_on=right_on, how='left')

''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''


def inner_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None):
    # join by common columns if nothing is specified
    if on == left_on == right_on == None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('inner joining on: ', common_cols)
        on = common_cols
        
    # check if on was specified and create left_on/right_on
    if on != None:
        left_on = right_on = on
        
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
    if on == left_on == right_on == None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('outer joining on: ', common_cols)
        on = common_cols
        
    # check if on was specified and create left_on/right_on
    if on != None:
        left_on = right_on = on
        
    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')
        
    return dataframe1.merge(dataframe2, left_on=left_on, right_on=right_on, how='outer')

''' _______________________________________________________________________________________________
	_______________________________________________________________________________________________ 
'''

def semi_join(dataframe1, dataframe2, on=None, left_on=None, right_on=None):
    # join by common columns if nothing is specified
    if on == left_on == right_on == None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('semi joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on != None:
        left_on = right_on = on
        
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
    if on == left_on == right_on == None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('anti joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on != None:
        left_on = right_on = on
        
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
    if on == left_on == right_on == None:
        print('No columns specified')
        common_cols = list(set(dataframe1.columns).intersection(dataframe2.columns))
        if len(common_cols) == 0:
            raise ValueError('No common columns exist')
        print('semi joining on: ', common_cols)
        on = common_cols

    # check if on was specified and create left_on/right_on
    if on != None:
        left_on = right_on = on

    if not all([col in dataframe1.columns for col in left_on]):
        raise ValueError('All columns not present in dataframe1')
    if not all([col in dataframe2.columns for col in right_on]):
        raise ValueError('All columns not present in dataframe2')

    dataframe2['_jointag_'] = 'tag'  # add a tag in case of joining on all columns (if all cols no NA to drop)
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

import numpy as np
from helperFunctions.writing import write_json


def weight_feature_by_freq(df, feature, add_wghts_to_df=False, add_counts_to_df=False, outfile=None):

    """
    function to calculate the sampling weighting for a feature such
    that each unique values will be selected with the same probability

    :param df: dataframe
    :param feature: feature with which to weight
    :param add_to_df: whether to return dataframe with new weight column
    :param outfile: name of file in which to save the count dictionary
    :return: either a df with the weights as a new column or a numpy array
    """

    df = df.copy()
    count_col = feature + '_counts'
    df[count_col] = np.array(df.groupby(feature)[feature].transform('count'))
    counts = df[count_col]
    weights = np.array(1 / (df[count_col] * df[feature].nunique()))

    if outfile is not None:
        # save count dictionary to look up values for updating
        df_nodup = df.drop_duplicates([feature, count_col])
        #  str needed for serializability in writing json file
        count_dic = {str(val): int(c) for val, c in zip(df_nodup[feature], df_nodup[count_col])}
        write_json(count_dic, outfile)

    if not add_counts_to_df:
        df.drop(count_col, axis=1, inplace=True)

    if add_wghts_to_df or add_counts_to_df:
        df[feature + '_weights'] = weights
        return df
    return weights, counts


def weight_target_by_freq(df, id_col, target_col, subtract_self=False, self_counts=None, outfile=None):

    """
    function to weight targets, only works when targets are values that share link
    not a very general function but much faster because of this.
    :param df:
    :param id_col:
    :param target_col:
    :param subtract_self:
    :param self_counts:
    :param outfile:
    :return:
    """
    df = df.copy()
    # determine number of times user shows up in target_user_indxs
    df['targ_count'] = df[target_col].apply(len)
    df['times_id_could_be_targ'] = df.groupby(id_col).targ_count.transform('sum')

    # must subtract number of times user had bet bet from this number to obtain
    # the possible times since a user can not be in target pool of own bet
    if self_counts is not None and subtract_self:
        df['times_id_could_be_targ'] -= self_counts

    target_counts = df[[id_col, 'times_id_could_be_targ']].drop_duplicates(id_col)

    # make dictionary from unique dataframe to save computation time
    itc_dic = {str(val): int(c) for val, c in
               zip(target_counts[id_col], target_counts.times_id_could_be_targ)}

    if outfile is not None:
        # write count dictionary for use in updating
        write_json(itc_dic, outfile)

    target_weights = df[target_col] \
        .apply(lambda x: np.array([1 / itc_dic[str(i)] for i in x]))

    return np.array(target_weights)

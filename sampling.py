import random
import numpy as np
import pandas as pd


def down_sample(df, downsample_rate, dependent_var = 'y', inplace=False):
    print('down sampling negative examples')
    df_neg = df[df[dependent_var] == 0]
    sampled_indices = random.sample(
        set(df_neg.index.values), int(round((1-downsample_rate)*len(df_neg))))
    if not inplace:
        return df.drop(sampled_indices, inplace=inplace)
    else:
        df.drop(sampled_indices, inplace=inplace)


def downsample_recal(predicted_probs, downsamp_rate):
    if isinstance(predicted_probs, (list)):
        predicted_probs = np.array(predicted_probs)
    recal_probs = predicted_probs/(predicted_probs + (1 - predicted_probs)/downsamp_rate)
    return recal_probs


def k_fold_sampling(df, k, target=None, stratified=False):

    k = int(k)

    if stratified and target == None:
        raise ValueError('Target variable is required for stratified sampling.')

    if stratified:
        # create list containing a dataframe for each target class
        targ_classes = df[target].unique()
        n_strata = len(targ_classes)
        stratified_dfs = [df[df[target] == t] for t in targ_classes]
        # perform unstratified k fold sampling on each dataframe in list
        sampled_data_frames = [k_fold_sampling(d, k, stratified=False) for d in stratified_dfs]
        # sampled_data_frames is a list of lists of dataframes
        # sampled_data_frames[i][j] gives a data frame of strata i in fold j

        # concatenate resulting dfs of same strata together for each fold
        final_df_list = [
            pd.concat([sampled_data_frames[i][j] for i in range(n_strata)], axis=0)
            for j in range(k)]

        return final_df_list
    else:
        # first shuffle the dataframe to randomly select
        shuffled_indx = np.random.permutation(df.index)
        df_shuf = df.reindex(shuffled_indx)
        # find the base number for each fold
        N = len(df_shuf)
        base_n = round(N/k)

        left_over = N - base_n*(k)
        # add or subtract from each fold to achieve full coverage
        if left_over >= 0:
            n_fold = [base_n]*(k-left_over) + [base_n + 1]*left_over
        else:
            n_fold = [base_n]*(k+left_over) + [base_n - 1]*abs(left_over)

        indices = list(np.cumsum(n_fold))
        low_indx = [0] + indices[:-1]
        high_indx = indices
        # return chunks of randomized dataframe
        samples = [df_shuf[low:high] for low, high in zip(low_indx, high_indx)]
        return samples



# testdf = pd.read_csv('/Users/keith.landry/data/testDf.csv')
# w = k_fold_sampling(testdf, 13)
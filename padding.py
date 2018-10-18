import pandas as pd
from helperFunctions.joinHelpers import left_join, outer_join

def pad_dates_by_group(df, group_col, date_col, fill_method='ffill', use_full_date_range=True):

    if use_full_date_range:
        all_dates = pd.date_range(df[date_col].min(), df[date_col].max())

        date_combos = df[[group_col]] \
            .groupby(group_col) \
            .apply(lambda x: x.reindex(pd.DatetimeIndex(all_dates))) \
            .drop(group_col, axis=1)

        new_index_names = np.where(
            [n is not None for n in date_combos.index.names], date_combos.index.names, date_col)

        # change index names (i.e. column names) so that they match
        date_combos.index.rename(new_index_names, inplace=True)
        date_combos.reset_index(inplace=True)  # two separate steps to reset column name

    else:
        date_combos = df[[group_col, date_col]] \
            .groupby(group_col) \
            .apply(lambda x: x.set_index(date_col).resample('D').count()) \
            .drop(group_col, axis=1) \
            .reset_index()

    # join to account for possibility of multiple entries per date per group in original DateFrame
    # don't sort here because it is already sorted by time going into the padding process
    padded_df = left_join(date_combos, df)

    padded_df = padded_df.groupby(group_col) \
        .apply(lambda x: x.fillna(method=fill_method))

    return padded_df


def pad(df, date_col, offset_string):
    
    all_dates = df.set_index(date_col).resample(offset_string).count().reset_index()[[date_col]]
    return outer_join(df, all_dates).sort_values(date_col)

    
    

    
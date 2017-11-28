import pandas as pd


def convert_to_datetime(df, columns):
    df = df.copy(deep=True)
    df[columns] = df[columns].apply(lambda col: pd.to_datetime(col, unit='ms'), axis=1)
    return df


def group_agg_pivot_df(df, group_cols, agg_func='count', agg_col=None):

    if agg_col is None:
        agg_col = group_cols[0]

    grouped = df.groupby(group_cols).agg({agg_col: agg_func})
    # levels to unstack. Don't include the first level since
    # we want the to keep identifying index
    unstack_lvls = list(range(len(group_cols)))[1:]
    grouped = grouped.unstack(level=unstack_lvls).fillna(0)
    grouped.columns = grouped.columns.droplevel()

    if len(group_cols) > 2:
        col_strs = [[str(c) for c in full_col] for full_col in grouped.columns.values]
        grouped.columns = ['_'.join(c).strip() for c in col_strs]

    pivot_df = pd.DataFrame(grouped.to_records())
    pivot_df.columns.values[1:] = [
        s.replace(' ', '_').lower() for s in pivot_df.columns.values[1:]]
    return pivot_df
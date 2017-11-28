import pandas as pd
import helperFunctions.sampling as samp

def train_kfold_models(train_data, target, model, k):

    # separate negative examples for down sampling
    neg_examples = train_data[train_data[target] == 0]
    pos_examples = train_data[train_data[target] == 1]
    del train_data

    # get k samples of
    k_fold_neg_dfs = samp.k_fold_sampling(neg_examples, k)
    del neg_examples

    model_list = []
    for df_neg in k_fold_neg_dfs:
        # add positive examples back in
        df_full = pd.concat([df_neg, pos_examples])
        mdl = model.fit(df_full.drop(target, axis=1), df_full[target])
        model_list.append(mdl)
        del df_full, df_neg

    return model_list

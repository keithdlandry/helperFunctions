# functions to help with plotting commonly used plots
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from helperFunctions.sampling import down_sample, downsample_recal
import random
import numpy as np
from cycler import cycler
from scipy.stats import gaussian_kde
from math import floor


def determine_grid_pos(x, y, n):

    # n starting at 0. So n=5 is actually the 6th
    j = n%y
    i = n//y
    return i, j


def determine_dimensions(n):

    sqrt_n = np.sqrt(n)
    floort_n = floor(sqrt_n)

    if sqrt_n == floort_n:
        x = y = floort_n
    else:
        x = floort_n
        y = floort_n + 1

    if x*y >= n:
        return x, y
    else:
        return x+1, y


def scatter_plot_classes(X, class_array, label_dict=None, title=None, alpha=.5,
                         legend_loc='best', figsize=(8, 6), color_cycl=None,
                         xlab=None, ylab=None, **kwargs):

    """
    :param X: 2D array (shape npoints, 2) of x-y values to be plotted.
    :param class_array: Array containing the class value to which each x-y point belongs
    :param label_dict: Dictionary of class value and legend label.
    :param title: Tittle of plot
    :param alpha:
    :param legend_loc:
    :param figsize:
    :param color_cycl: A list of colors to be used for color selection of each class
    :param kwargs:
    :return: Returns matplotlib figure.
    """
    unique_vals = class_array.unique()
    masks = [class_array == u for u in unique_vals]

    f, ax = plt.subplots(figsize=figsize)
    if color_cycl is not None:
        ax.set_prop_cycle(cycler('color', color_cycl))

    for m, u in zip(masks, unique_vals):
        ax.scatter(X[m, 0], X[m, 1], alpha=alpha,
                   label=label_dict[u], **kwargs)
    ax.legend(loc=legend_loc)

    if title is not None:
        ax.set_title(title)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)

    plt.show()
    return f


def scatter_facet(X, class_array, label_dict=None, title=None, alpha=.5,
                  legend_loc='best', figsize=(8, 6), xlab=None, ylab=None, **kwargs):

    """
    :param X: 2D array (shape npoints, 2) of x-y values to be plotted.
    :param class_array: Array containing the class value to which each x-y point belongs
    :param label_dict: Dictionary of class value and legend label.
    :param title: Tittle of plot
    :param alpha:
    :param legend_loc:
    :param figsize:
    :param kwargs:
    :return: Returns matplotlib figure.
    """

    unique_vals = class_array.unique()
    masks = [class_array == u for u in unique_vals]

    a, b = determine_dimensions(len(unique_vals))

    f, ax = plt.subplots(a, b, figsize=figsize)

    for m, u, k in zip(masks, unique_vals, range(len(unique_vals))):

        i, j = determine_grid_pos(a, b, k)

        ax[i][j].scatter(X[m, 0], X[m, 1], alpha=alpha,
                         label=label_dict[u], ** kwargs)
        ax[i][j].legend(loc=legend_loc)

    for i in range(a):
        ax[i][0].set_ylabel(ylab)
    for i in range(b):
        ax[b-1][i].set_xlabel(xlab)

    if title is not None:
        plt.suptitle(title)
    plt.show()
    return f


def calibration_plot(g, xvar, yvar, n_pos, xmin=None, xmax=None,
                     ymin=None, ymax=None, title=None, background_rate=None):
    '''
    :param g: dataframe with columns
    :param xvar: column name of g containing the x variable to be plotted
        (typically the mean predicted probability of group
    :param yvar: column name of g containing the y variable to be plotted
        (typically percentage of positive examples for group)
    :param n_pos: column name of g containing the number of positive examples in group
    :param xmin: minimum of x axis of plot
    :param xmax: maximum of x axis of plot
    :param ymin: minimum of y axis of plot
    :param ymax: maximum of y axis of plot
    :param title: title of plot
    :param background_rate: percentage of positive examples in whole data set
    :return: returns matplot lib figure
    '''

    if xmin is None:
        if xmax is None:
            xmin = g[xvar].min() - (g[xvar].max() - g[xvar].min())/25
        else:
            xmin = g[xvar].min() - (xmax - g[xvar].min())/25
        print('xmin =', xmin)
    if xmax is None:
        if xmin is None:
            xmax = g[xvar].max() + (g[xvar].max() - g[xvar].min())/25
        else:
            xmax = g[xvar].max() + (g[xvar].max() - xmin)/25
        print('xmax =', xmax)
    if ymin is None:
        ymin = g[yvar].min() - .001
    if ymax is None:
        ymax = g[yvar].max() + .001

    f, ax = plt.subplots(2, 1, figsize=(8, 6))
    f.subplots_adjust(hspace=0)
    ax[0].axhline(y=background_rate, linewidth=1, color='grey', linestyle=':', zorder=1)
    ax[0].scatter(g[xvar], g[yvar], zorder=3)

    n = g[n_pos]/g[yvar]
    yerr = np.sqrt(g[yvar]*(1-g[yvar])/n)
    ax[0].errorbar(g[xvar], g[yvar], yerr=yerr, fmt='o')
    # plot y=x line
    ax[0].plot(
        ax[0].get_xlim(), ax[0].get_xlim(), linewidth=1, linestyle='--', color='black', zorder=2)
    ax[0].get_xaxis().set_visible(False)
    # ax[0].set_xlabel('model click probability')
    ax[0].set_ylabel('True CTR in bin')
    if xmin is not None or xmax is not None:
        ax[0].set_xlim(xmin, xmax)
    if ymin is not None or ymax is not None:
        ax[0].set_ylim(ymin, ymax)
    if title is not None:
        ax[0].set_title(title)

    # need to remove any rows in g with <= zero positive examples because log won't work
    g = g[g[n_pos] > 0]
    ax[1].scatter(g[xvar], g[n_pos], color='black')
    ax[1].set_xlabel('model click probability')
    ax[1].set_ylabel('number of clicks in bin')
    #ax[1].set_ylim(g[n_pos].min(), g[n_pos].max() + 100)
    #ax[1].set_ylim(10**-1, 10**3.5)
    ax[1].set_yscale('log')
    ax[1].set_xlim(ax[0].get_xlim())
    print(g[n_pos].min(), g[n_pos].max())
    print(ax[1].get_ylim())

    plt.show()
    return f


def plot_learning_curve(model, train_data, test_data, target, metric_funct, y_axis_label,
                        data_fracs=[0.01, 0.1, 0.25, 0.5, 0.75, 1], n_iter=10,
                        predict_class=False, classification=True,
                        xlim=None, ylim=None, figsize=(8,6),  **kwargs):

    train_performance = []
    test_performance = []
    train_fraction = []
    for data_frac in data_fracs:

        sample_perf_train = []
        sample_perf_test = []
        train_fraction.append(data_frac)
        for i in range(n_iter):

            train_data.sample(frac=data_frac)
            model.fit(train_data.drop(target, axis=1), train_data[target])

            if classification and not predict_class:
                pred_test = model.predict_proba(test_data.drop(target, axis=1))
                pred_train = model.predict_proba(train_data.drop(target, axis=1))
            else:
                pred_test = model.predict(test_data.drop(target, axis=1))
                pred_train = model.predict(train_data.drop(target, axis=1))

            sample_perf_test.append(metric_funct(test_data[target], pred_test, **kwargs))
            sample_perf_train.append(metric_funct(train_data[target], pred_train, **kwargs))

        train_performance.append(np.mean(sample_perf_train))
        test_performance.append(np.mean(sample_perf_test))

    f, ax = plt.subplots(figsize=figsize)
    ax.plot(train_fraction, test_performance, 'o-', label='Test Set Performance')
    ax.plot(train_fraction, train_performance, 'o-', label='Train Set Performance')
    ax.set_xlabel('Training Fraction')
    ax.set_ylabel(y_axis_label)
    if 'background_rate' in kwargs:
        ax.axhline(y=1, linewidth=1, color='grey', linestyle=':', zorder=1, label='positive rate')
    ax.legend(loc='best')
    return f


def plot_learning_curve2(model, train_data, target, scoring, y_axis_label,
                         cv=None, train_sizes=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
                         xlim=None, ylim=None, figsize=(8,6), n_jobs=1,  **kwargs):

    train_sizes, train_scores, test_scores = learning_curve(
        model, train_data.drop(target, axis=1), train_data[target],
        cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    #train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #test_scores_std = np.std(test_scores, axis=1)

    f, ax = plt.subplots(figsize=figsize)
    ax.plot(train_fraction, test_scores_mean, 'o-', label='Test Set Performance')
    ax.plot(train_fraction, train_scores_mean, 'o-', label='Train Set Performance')
    ax.set_xlabel('Training Fraction')
    ax.set_ylabel(y_axis_label)
    if 'background_rate' in kwargs:
        ax.axhline(y=1, linewidth=1, color='grey', linestyle=':', zorder=1, label='positive rate')
    ax.legend(loc='best')
    return f


def log_reg_learning_curve(df, target_var,
                           downsamp_rate=None,
                           background_rate=None,
                           random_samples=True,
                           samp_fracs=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
                           n_iter=1):

    train_data, test_data = train_test_split(df, test_size=0.2)

    if downsamp_rate is not None:
        train_data = down_sample(train_data, ds_rate, target_var, inplace=False)

    train_size = []
    performance = []

    for samp_frac in samp_fracs:

        n_train_ex = round(samp_frac*len(train_data))
        train_size.append(samp_frac)

        sample_perf = []
        for i in range(n_iter):
            if random_samples:
                sampled_ind = random.sample(set(train_data.index.values), n_train_ex)
                sampled_data = train_data.loc[sampled_ind]
            else:
                sampled_data = train_data[:n_train_ex]

            Y_train = sampled_data[target_var]
            X_train = sampled_data.drop(target_var, axis=1)
            Y_test = test_data[target_var]
            X_test = test_data.drop(target_var, axis=1)

            # cv log reg
            lrcv_mdl = LogisticRegressionCV(Cs=10, cv=2, random_state=12)
            lrcv_mdl.fit(X_train, Y_train)
            lrcv_pred = lrcv_mdl.predict_proba(X_test)[:, 1]
            lrcv_pred = downsample_recal(lrcv_pred, ds_rate)
            x_entr = log_loss(Y_test, lrcv_pred)

            if background_rate is not None:
                bg_log_loss = log_loss(Y_test, [background_rate]*len(Y_test))
                sample_perf.append(x_entr/bg_log_loss)
            else:
                sample_perf.append(x_entr)

        performance.append(np.mean(sample_perf))
        print('\ntraining fraction =', samp_frac)
        if background_rate is not None:
            print('normalized log loss =', np.mean(sample_perf))
        else:
            print('log loss =', np.mean(sample_perf))

    f, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(train_size, performance)
    ax.set_xlabel('Training Fraction')
    ax.set_ylabel('Log Loss')
    if background_rate is not None:
        ax.axhline(y=1, linewidth=1, color='grey', linestyle=':', zorder=1)
    plt.show()


"""
Try to use threading. Didn't seem to work
"""

#
# import threading
# from multiprocessing.dummy import Pool
# from functools import partial
#
#
# def log_reg_learning_curve(df, target_var,
#                            downsamp_rate=None,
#                            background_rate=None,
#                            random_samples=True,
#                            samp_fracs=[0.01, 0.1, 0.25, 0.5, 0.75, 1],
#                            n_iter=1):
#
#     train_data, test_data = train_test_split(df, test_size=0.2)
#
#     if downsamp_rate is not None:
#         train_data = down_sample(train_data, ds_rate, target_var, inplace=False)
#
#     train_size = []
#     performance = []
#
#     for samp_frac in samp_fracs:
#
#         train_size.append(samp_frac)
#
#         # parallelize n_iter calculations of
#         # the log loss
#         with Pool() as pool:
#             res = pool.map(
#                 partial(
#                     func1,
#                     train_data=train_data,
#                     test_data=test_data,
#                     target_var=target_var,
#                     ds_rate=ds_rate,
#                     background_rate=background_rate,
#                     random_samples=random_samples
#                 ),
#                 [samp_frac]*n_iter
#             )
#         sample_perf = np.mean(res)
#
#         performance.append(sample_perf)
#         print('\ntraining fraction =', samp_frac)
#         if background_rate is not None:
#             print('normalized log loss =', np.mean(sample_perf))
#         else:
#             print('log loss =', np.mean(sample_perf))
#
#     f, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(train_size, performance)
#     ax.set_xlabel('Training Fraction')
#     ax.set_ylabel('Log Loss')
#     if background_rate is not None:
#         ax.axhline(y=1, linewidth=1, color='grey', linestyle=':', zorder=1)
#     plt.show()
#
#
# def func1(samp_frac, train_data, test_data, target_var, ds_rate, background_rate=None, random_samples=True):
#
#         n_train_ex = round(samp_frac*len(train_data))
#
#         if random_samples:
#             sampled_ind = random.sample(set(train_data.index.values), n_train_ex)
#             sampled_data = train_data.loc[sampled_ind]
#         else:
#             sampled_data = train_data[:n_train_ex]
#
#         Y_train = sampled_data[target_var]
#         X_train = sampled_data.drop(target_var, axis=1)
#         Y_test = test_data[target_var]
#         X_test = test_data.drop(target_var, axis=1)
#
#         # cv log reg
#         lrcv_mdl = LogisticRegressionCV(Cs=10, cv=2, random_state=12)
#         lrcv_mdl.fit(X_train, Y_train)
#         lrcv_pred = lrcv_mdl.predict_proba(X_test)[:, 1]
#         lrcv_pred = downsample_recal(lrcv_pred, ds_rate)
#         x_entr = log_loss(Y_test, lrcv_pred)
#
#         if background_rate is not None:
#             bg_log_loss = log_loss(Y_test, [background_rate] * len(Y_test))
#             return x_entr / bg_log_loss
#
#         return x_entr


def add_jitter(x):
    kde = gaussian_kde(x)
    jitter = np.random.rand(len(x)) -.5
    xvals = x + (density * jitter)
    plt.scatter(xvals, y)
    plt.scatter(x,y)
    plt.show()


def add_jitter(x, jitter=.1):

    delta = np.max(x) - np.min(x)
    offset = np.random.rand(len(x)) - .5
    x = x + jitter*offset*delta
    return x


def jitter_plot(x, y, jitter_x=.1, jitter_y=.1, **kwargs):
    if jitter_x > 0:
        x = add_jitter(x, jitter_x)
    if jitter_y > 0:
        y = add_jitter(y, jitter_y)

    return plt.scatter(x, y, **kwargs)


def plot_function(function, x_range, **kwargs):
    x = np.array(x_range)
    y = function(x)
    plt.plot(x, y, **kwargs)

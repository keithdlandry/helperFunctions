import pickle
import datetime
import warnings
import json
import pandas as pd
import sys_utils.s3communicator
from helperFunctions.printing import print_color


def write_model_metadata(model_file_name, model, path, model_name, feat_names=None, train_file=None):

    path = path + '/' if path[-1] != '/' else path
    file_name = path + model_name + '.txt'
    print_color('cyan', 'Writing Model Metadata To: {}'.format(file_name))

    with open(file_name, 'w') as f:
        f.write('Model File:\t')
        f.write(model_file_name)
        f.write('\n\n')

        if train_file is not None:
            f.write('Trained On:\t')
            f.write(train_file)
            f.write('\n\n')

        f.write('Model Parameters:\n\n')
        try:
            f.write(json.dumps(model.__dict__, indent=4))
        except TypeError as err:
            print('TypeError: {}'.format(err))
            print('It will not be printed with model parameters.')
            try:
                f.write(json.dumps(model.get_params(), indent=4))
            except AttributeError:
                print("Could't get model parameters.")
        f.write('\n\n')

        if feat_names is None:
            try:
                f.write('Feature Importances:\n\n')
                f.write(json.dumps(model.feature_importances_.tolist(), indent=4))
            except AttributeError:
                print('Model does not have feature importance')
            # feat importance doesn't exist in model
        else:
            try:
                f.write('Feature Importances:\n\n')
                imps = pd.DataFrame({'feature': feat_names,
                                     'importance': model.feature_importances_})
                imps.sort_values('importance', inplace=True)
                f.write(repr(imps))
            except AttributeError:
                print('Model does not have feature importance')
            # feat importance doesn't exist in model

    print('Metadata Written')


def write_model(model, path, model_name, feat_names=None, train_file=None
                , add_timestamp=True, save_matadata=False):

    path = path + '/' if path[-1] != '/' else path
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    unique_id = '_{}'.format(time_stamp) if add_timestamp else ''

    model_name = model_name + '{}'.format(unique_id)
    file_name = path + model_name + '.pickle.dat'
    print_color('cyan', 'Writing Model To: {}'.format(file_name))
    pickle.dump(model, open(file_name, "wb"))

    if save_matadata:
        if feat_names is None:
            warnings.warn('Feature names not passed. Model metadata will not be saved.')
        write_model_metadata(file_name, model, path, model_name, feat_names, train_file)


def write_to_s3(df, filename, bucket='geniussports-machine-learning-data'):

    s3c = sys_utils.s3communicator.S3Communicator()
    s3c.put_file(df, bucket, filename)






import s3fs
import pandas as pd
import sys_utils.io_utils as io_utils
import feather
import json


def load_from_s3(bucket, key, aws_profile, **kwargs):
    path = 's3://{buck}/{key}'.format(buck=bucket, key=key)
    s3file_sys = s3fs.S3FileSystem(profile_name=aws_profile)  # pass profile name as kwarg to boto3

    with s3file_sys.open(path) as f:
        return pd.read_csv(f, **kwargs)  # compression='gzip', sep='|' (most of the time)


def generic_load(file_name, file_type=None, json_gz=False):

    print('loading file', file_name)

    if file_type is None:
        file_type = file_name[file_name.rfind('.')+1:]

    if file_type == 'csv':
        data = pd.read_csv(file_name)
    elif file_type == 'feather':
        data = feather.read_dataframe(file_name)
    elif file_type == 'json':
        data = io_utils.from_jsonz(file_name, add_gz=json_gz)
    else:
        raise TypeError('Unrecognized file type')
    return data


def load_raw_json(file_name):
    return json.load(open(file_name))





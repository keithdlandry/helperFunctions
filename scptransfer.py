import os
from helperFunctions.printing import print_color


def scp_transfer(local_file, remote_file, user='ec2-user', host_ip='ec2-54-187-143-94',
                 host_suffix='.us-west-2.compute.amazonaws.com', to_host=False,
                 key_file='~/aws_keys/machine-learning-devel-tca.pem'):

    if not to_host:
        scp_string = 'scp -i {} {}@{}{}:{} {}'\
            .format(key_file, user, host_ip, host_suffix, remote_file, local_file)
    else:
        scp_string = 'scp -i {} {} {}@{}{}:{}'\
            .format(key_file, local_file, user, host_ip, host_suffix, remote_file)

    print_color('magenta', scp_string)

    os.system(scp_string)

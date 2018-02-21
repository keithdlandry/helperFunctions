import warnings


def print_color(color, text, *args):

    colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

    color_dict = {c: 30+n for n, c in enumerate(colors)}
    bright_color_dict = {'bright_{}'.format(c): 90+n for n, c in enumerate(colors)}

    color_dict.update(bright_color_dict)

    if color not in color_dict.keys():
        warnings.warn('\nWarning from print_color:\n'
                      'Unrecognized color, printing will be default. \n'
                      'Recognized colors are: \n'
                      '{}'.format(list(color_dict.keys())))

        print(text, *args)
    else:
        csi = '\x1B[{}m'.format(color_dict[color])
        csi_reset = '\x1B[0m'
        print(csi, end='')  # start the color
        print(text, *args)
        print(csi_reset, end='')  # reset to original color

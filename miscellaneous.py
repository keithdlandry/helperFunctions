def list_to_comma_sep(mylist):
    mylist = [str(l) for l in mylist]
    mylist = ','.join(mylist)
    return mylist


def make_list_if_not_list(input):
    if not isinstance(input, (list)):
        return [input]
    return input


def get_dict_combinations(dic, index=0):
    # Recursive function to generate dictionaries of all combinations of values.
    # Because of this, index should not be changed by hand!
    all_dicts = []
    new_dict = {}

    keys = list(dic.keys())
    key = keys[index]

    for val in make_list_if_not_list(dic[key]):
        new_dict[key] = val
        if index < len(keys) - 1:
            # move to next parameter (index + 1)
            # get all combinations of parameters [1:]
            # this is done by recursion
            for d in get_dict_combinations(dic, index=index + 1):
                ndic = dict(list(new_dict.items()) + list(d.items()))
                all_dicts.append(ndic)
        else:
            # on last parameter, append to the list of dictionaries
            all_dicts.append(new_dict.copy())  # copy is needed so the dictionary isn't altered on further recursions
    return all_dicts


def strip_suffix(s, suffix):
    if s.endswith(suffix):
        return s[:len(s) - len(suffix)]
    return s

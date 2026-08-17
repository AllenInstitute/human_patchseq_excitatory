# This module is from https://github.com/AllenInstitute/mouse-met-figures

def sum_directed_edges(melted):
    undir_pair_dict = {}
    undir_edge_list = []
    for r in melted.itertuples():
        if r.assigned_met == r.other_met:
            continue

        dict_key = frozenset([r.assigned_met, r.other_met])
        if dict_key not in undir_pair_dict:
            undir_pair_dict[dict_key] = r.value
        else:
            undir_pair_dict[dict_key] += r.value

    for k in undir_pair_dict:
        a, b = tuple(k)
        undir_edge_list.append((
            a,
            b,
            {"weight": undir_pair_dict[k],
            }
        ))

    return undir_edge_list


def met_tuple_to_str(name):
    return str(name[0]) + "-" + str(name[1])


def met_str_to_tuple(name):
    split_name = name.split("-")
    return (int(split_name[0]), split_name[1])

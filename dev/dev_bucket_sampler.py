from collections import OrderedDict

import numpy as np

batch_size = 5
samples = 33
min_len = 6
max_len = 14

lengths = np.random.randint(min_len, max_len, size=samples)
indices = list(range(len(lengths)))
idx_len_list = list(zip(indices, lengths))

for i in range(2):
    print(i)

    len_map = OrderedDict()
    for i, l in idx_len_list:
        if l not in len_map:
            len_map[l] = [i]
        else:
            len_map[l].append(i)

    suffled_grouped_indices = []
    for l, indices in reversed(sorted(len_map.items())):
        np.random.shuffle(indices)
        suffled_grouped_indices.extend(indices)

    for group in [suffled_grouped_indices[i:(i+batch_size)] for i in range(0, len(lengths), batch_size)]:
        print(group)

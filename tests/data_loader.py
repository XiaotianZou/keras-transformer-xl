import numpy as np
import allensdk
from allensdk.core.nwb_data_set import NwbDataSet
import os


def get_all_file(path):
    result = []
    target_folder = os.walk(path)
    for path, dir_list, file_list in target_folder:
        print(dir_list)
        for dir_name in dir_list:
            dir_path = os.path.join(path, dir_name)
            for tmp_path, tmp_dir_list, tmp_file_list in os.walk(dir_path):
                for file_name in tmp_file_list:
                    if file_name.find('ephy') != -1:
                        result.append(os.path.join(tmp_path, file_name))
    return result

file_list = get_all_file('../../share/quonlab/data/allen/')

print(file_list)        



# sweep_ids = data_set.get_sweep_numbers()

# sweep_data_arr = []

# for sweep_id in sweep_ids:
#   sweep_data = data_set.get_sweep(sweep_id)
#   sweep_data_arr.append(sweep_data)

# preprocess_sti, preprocess_res, split_mark = [], [], []
# split_mark.append(0)
# for sweep_data in sweep_data_arr:
#   index_range = sweep_data['index_range']
#   i = sweep_data['stimulus'][:index_range[1] + 1]
#   v = sweep_data['response'][:index_range[1] + 1]
#   preprocess_sti.extend(i)
#   preprocess_res.extend(v)
#   split_mark.append(len(preprocess_res))

# sti_arr = [preprocess_sti[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)]
# res_arr = [preprocess_res[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)]

# sti_arr = np.array(preprocess_sti)
# res_arr = np.array(preprocess_res)
# sti_arr = sti_arr.flatten()
# res_arr = res_arr.flatten()

# sti_arr = sti_arr.reshape(-1, 1)
# res_arr = res_arr.reshape(-1, 1)

# scaler_sti = StandardScaler().fit(sti_arr)
# sti_arr = scaler_sti.transform(sti_arr)
# scaler_res = StandardScaler().fit(res_arr)
# res_arr = scaler_res.transform(res_arr)

# sti_arr = sti_arr.flatten()
# res_arr = res_arr.flatten()

# sti_arr = sti_arr.tolist()
# res_arr = res_arr.tolist()

# sti_arr = [sti_arr[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)]
# res_arr = [res_arr[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)]

# train_sti = np.array(train_sti)
# train_res = np.array(train_res)
# print(train_res[0].shape)
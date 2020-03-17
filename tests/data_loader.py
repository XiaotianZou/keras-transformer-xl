import numpy as np
import allensdk
from allensdk.core.nwb_data_set import NwbDataSet
import os
from sklearn.preprocessing import *

print('hello world!')
def get_all_file(path):
    result = []
    target_folder = os.walk(path)
    for path, dir_list, file_list in target_folder:
        for dir_name in dir_list:
            dir_path = os.path.join(path, dir_name)
            for tmp_path, tmp_dir_list, tmp_file_list in os.walk(dir_path):
                for file_name in tmp_file_list:
                    if file_name.find('nwb') != -1:
                        result.append(os.path.join(tmp_path, file_name))
    return result

file_list = get_all_file('/share/quonlab/data/allen/')

data_set = NwbDataSet(file_list[1])

sweep_ids = data_set.get_sweep_numbers()

sweep_data_arr = []

for sweep_id in sweep_ids:
  sweep_data = data_set.get_sweep(sweep_id)
  sweep_data_arr.append(sweep_data)

preprocess_sti, preprocess_res, split_mark = [], [], []
split_mark.append(0)
for sweep_data in sweep_data_arr:
  index_range = sweep_data['index_range']
  i = sweep_data['stimulus'][:index_range[1] + 1]
  v = sweep_data['response'][:index_range[1] + 1]
  preprocess_sti.extend(i)
  preprocess_res.extend(v)
  split_mark.append(len(preprocess_res))

preprocess_sti = np.array(preprocess_sti)
preprocess_sti = preprocess_sti.reshape(-1, 1)
preprocess_res = np.array(preprocess_res)
preprocess_res = preprocess_res.reshape(-1, 1)


sti_arr = [preprocess_sti[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)]
res_arr = [preprocess_res[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)]

sti_diff_arr = np.array([np.diff(sti_arr[i], axis=0) for i in range(len(sti_arr))])
res_diff_arr = np.array([np.diff(res_arr[i], axis=0) for i in range(len(res_arr))])


split_mark = [0]
size = 0
for i in range(len(sti_diff_arr)):
  size += len(sti_diff_arr[i])
  split_mark.append(size)

sti_diff_arr_flat, res_diff_arr_flat = [], []
for i in range(len(sti_diff_arr)):
  if i == 0:
    sti_diff_arr_flat = sti_diff_arr[i]
    res_diff_arr_flat = res_diff_arr[i]
  else:
    sti_diff_arr_flat = np.concatenate((sti_diff_arr_flat, sti_diff_arr[i]))
    res_diff_arr_flat = np.concatenate((res_diff_arr_flat, res_diff_arr[i]))


train_sti_diff = StandardScaler().fit_transform(sti_diff_arr_flat)
train_res_diff = StandardScaler().fit_transform(res_diff_arr_flat)

train_sti_diff = np.array([train_sti_diff[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)])
train_res_diff = np.array([train_res_diff[split_mark[i]: split_mark[i + 1]] for i in range(len(split_mark) - 1)])


for i in range(len(train_sti_diff)):
  train_sti_diff[i] = train_sti_diff[i].flatten()
  train_res_diff[i] = train_res_diff[i].flatten()

sti_time_step = 500
res_time_step = 100
pred_time_step = 30

training_sti_x = []
training_res_x = []
training_y = []

for k in range(len(train_sti_diff)):
  count = 0
  tmp_sti = np.array([])
  tmp_res = np.array([])
  tmp_y = np.array([])
  for i in range(sti_time_step, len(train_sti_diff[k]) - pred_time_step, pred_time_step):
    index = min(i, len(train_sti_diff[k]) - sti_time_step)
    index = i
    count += 1
    if (count == 1):
      tmp_sti = train_sti_diff[k][index - sti_time_step: index].reshape(1, 500)
      tmp_res = train_res_diff[k][index - res_time_step: index].reshape(1, 100)
      tmp_y = train_res_diff[k][index: index + pred_time_step].reshape(1, 30)
    else:
      tmp_sti = np.concatenate((tmp_sti, train_sti_diff[k][index - sti_time_step: index].reshape(1, 500)), axis=0)
      tmp_res = np.concatenate((tmp_res, train_res_diff[k][index - res_time_step: index].reshape(1, 100)), axis=0)
      tmp_y = np.concatenate((tmp_y, train_res_diff[k][index: index + pred_time_step].reshape(1, 30)), axis=0)
  training_sti_x.append(tmp_sti)
  training_res_x.append(tmp_res)
  training_y.append(tmp_y)
  count = 0
  tmp_sti = np.array([])
  tmp_res = np.array([])
  tmp_y = np.array([])
    # # 得到训练数据trainig_sti_x, training_res_x
    # training_sti_x.append(train_sti_diff[k][index - sti_time_step: index])
    # training_res_x.append(train_res_diff[k][index - res_time_step: index])
    # # 得到训练y数据training_y
    # y = train_res_diff[k][index: index + pred_time_step]
    # training_y.append(y)

training_res_x = np.array(training_res_x)
training_sti_x = np.array(training_sti_x)
training_y = np.array(training_y)
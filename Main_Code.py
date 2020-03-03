import numpy as np
import scipy.io

import os
import pandas as pd

from scipy.stats import invgamma, uniform
import scipy.stats as stats

import sys
# import skgarden as sg
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor



from utility import *
from p_class import *


def run_main(dataNum, dimNum, treeNum):

    ydata_train, xdata_train = data_generate_2(dataNum, dimNum)

    ydata_train_mean = np.mean(ydata_train)
    ydata_train = ydata_train - ydata_train_mean

    ydata_test, xdata_test = data_generate_2_true(dataNum, dimNum)
    ydata_test = ydata_test - ydata_train_mean

    min_num = 3

    tau_scaling_seq = (np.arange(9)+1)*0.1
    rmse_table = []
    leaf_num_table = []
    for tau_scaling in tau_scaling_seq:
        tau_seq = tau_scaling*(np.arange(dataNum)+1)**(1/(dimNum+1))

        bsp_forest = []
        for ti in range(treeNum):
            bsp_forest.append(bspf_str(min_num, dimNum))

        ylabel_forest = []
        ylabel_forest_test = []
        yval_predict = []
        for ti in range(treeNum):
            leaf_num = []
            label_seq = [0]
            label_per_data = np.array([], dtype=int)
            for ii in range(dataNum):
                [label_per_data, label_seq] = bsp_forest[ti].update_new(xdata_train[ii], xdata_train[:ii], label_per_data, label_seq, tau_seq[ii])
                if np.mod(ii, 100)==0:
                    print(ii)
                if len(label_per_data)!=(ii+1):
                    a = 1

            ylabel_forest.append(label_per_data)
            pd_val = pd.DataFrame(np.vstack((ylabel_forest[-1], ydata_train)).T, columns = ['label', 'val'])
            group_label_val = pd_val.groupby('label').mean().reset_index()
            group_label = group_label_val['label'].astype(int)
            group_val = group_label_val['val']

            ylabel_forest_test.append(bsp_forest[ti].cal_label(xdata_test))
            max_label = np.max([np.max(ylabel_forest[-1]), np.max(ylabel_forest_test[-1])])
            predict_dict = np.zeros(max_label)
            predict_dict[group_label-1] = group_val
            predict_test = predict_dict[np.asarray(ylabel_forest_test[-1])-1]
            yval_predict.append(predict_test)

            leaf_num.append(np.sum(np.asarray(bsp_forest[ti].treeStr)[:, 1]==-1))

        final_prediction = np.mean(np.asarray(yval_predict), axis=0)
        RMSE = np.sqrt(np.mean((final_prediction.reshape((-1)) - ydata_test.reshape((-1))) ** 2))
        rmse_table.append(RMSE)

        leaf_num_table.append(np.mean(leaf_num))
    path = ''

    np.savez_compressed(path+'Sine_comparison'+str(uniform.rvs()), rmse_table= rmse_table, leaf_num_table = leaf_num_table)
    # print('RMSE is: ', np.sqrt(np.mean((final_prediction.reshape((-1))-ydata_test.reshape((-1)))**2)))
    #



if __name__ == '__main__':

    dataNum = 5000
    dimNum = 2
    treeNum = 2
    run_main(dataNum, dimNum, treeNum)


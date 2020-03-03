import numpy as np
import scipy.io

import os
import pandas as pd
from scipy.stats import invgamma, uniform, norm, expon

from scipy.spatial import ConvexHull


import sys


def data_generate_2(dataNum, dimNum):
    xdata = uniform.rvs(size=(dataNum, dimNum))
    ydata = 10*np.sin(xdata[:, 0]*np.pi*xdata[:, 1])+0.2*norm.rvs(size=dataNum)

    return ydata, xdata

def data_generate_2_true(dataNum, dimNum):
    xdata = uniform.rvs(size=(dataNum, dimNum))
    ydata = 10*np.sin(xdata[:, 0]*np.pi*xdata[:, 1])

    return ydata, xdata


def data_generate(dataNum, dimNum):
    xdata = uniform.rvs(size=(dataNum, dimNum))
    ydata = 10*np.sin(xdata[:, 0]*np.pi*xdata[:, 1])+20*((xdata[:, 2]-0.5)**2)+10*xdata[:, 3]+5*xdata[:, 4]+norm.rvs(size=dataNum)

    return ydata, xdata

def data_generate_true(dataNum, dimNum):
    xdata = uniform.rvs(size=(dataNum, dimNum))
    ydata = 10*np.sin(xdata[:, 0]*np.pi*xdata[:, 1])+20*((xdata[:, 2]-0.5)**2)+10*xdata[:, 3]+5*xdata[:, 4]

    return ydata, xdata


def sequentialProcess_gen(sequentialPoints):

    rs_correct = True
    while (rs_correct):
        orthogonal_theta = np.random.rand()*np.pi
        positionB = np.array([np.cos(orthogonal_theta), np.sin(orthogonal_theta)])
        sequential_dis = np.sum(sequentialPoints*positionB, axis = 1)
        seq_max = np.max(sequential_dis)
        seq_min = np.min(sequential_dis)
        largeRatio = np.pi *np.sqrt(2)

        if (largeRatio*np.random.rand() <= (seq_max-seq_min)):
            rs_correct = False
    cut_position = seq_min*positionB+np.random.rand()*(seq_max-seq_min)*positionB
    cut_direction = np.array([-positionB[1], positionB[0]])

    point3 = cut_position
    point4 = cut_position + cut_direction

    return np.vstack((point3, point4))


def sequentialProcess_gen_no_crossing(cv_points_small, cv_points_large):

    rs_correct = True
    while (rs_correct):
        orthogonal_theta = np.random.rand()*np.pi
        positionB = np.array([np.cos(orthogonal_theta), np.sin(orthogonal_theta)])

        # for small one
        sequential_dis = np.sum(cv_points_small*positionB, axis = 1)
        seq_max_s = np.max(sequential_dis)
        seq_min_s = np.min(sequential_dis)


        # for large one
        sequential_dis = np.sum(cv_points_large*positionB, axis = 1)
        seq_max_l = np.max(sequential_dis)
        seq_min_l = np.min(sequential_dis)

        largeRatio = np.pi *np.sqrt(2)

        if (largeRatio*np.random.rand() <= ((seq_max_l-seq_min_l)-(seq_max_s-seq_min_s))):
            rs_correct = False

    if (seq_max_s==seq_max_l):
        cut_position = seq_min_l * positionB + np.random.rand() * (seq_min_s - seq_min_l) * positionB
    else:
        cut_position = seq_max_s * positionB + np.random.rand() * (seq_max_l - seq_max_s) * positionB

    cut_direction = np.array([-positionB[1], positionB[0]])

    point3 = cut_position
    point4 = cut_position + cut_direction

    return np.vstack((point3, point4))


def perimeter_cal(selected_points):
    d1 = np.sum(np.sum((selected_points[:-1]-selected_points[1:])**2, axis=1)**(0.5))
    d2 = (np.sum((selected_points[0]-selected_points[-1])**2)**(0.5))
    return d1 + d2

def update_cv(current_data, label_per_data, cu_index, index_p, dim_pair, KK, cv_points, cv_perimeters):
    # label_per_data[cu_index[index_p]] = KK
    if (np.sum(label_per_data == KK) > 2)&(len(np.unique(current_data[label_per_data == KK, 0]))>2):
        cv_points_new = []
        cv_perimeters_new = []
        new_points = current_data[label_per_data == KK]
        for dim_pair_i in (dim_pair):
            cv_str_new = ConvexHull(new_points[:, dim_pair_i])
            cv_points_new.append(new_points[cv_str_new.vertices][:, dim_pair_i])
            cv_perimeters_new.append(perimeter_cal(cv_points_new[-1]))  # points need to be index later
        cv_points.append(cv_points_new)
        cv_perimeters.append(cv_perimeters_new)
    else:
        cv_points.append([])
        cv_perimeters.append([])
    return cv_points, cv_perimeters


def CutConvexHull(label_seq, label_per_data, start_pos, dim_pair, current_data, cut_pos, tau_i, cut_line_points, treeStr, cv_points, cv_perimeters):

    if label_seq[start_pos] < 3:  # ensure the node has at least 3 data points
        # label_seq[start_pos] += 1
        # label_per_data = np.append(label_per_data, start_pos)
        return label_seq, label_per_data, cut_pos, cut_line_points, treeStr, cv_points, cv_perimeters

    # points_edge = []
    # periemeter_seq = np.zeros(len(dim_pair))
    # cut_indicator_p = current_data[label_per_data == start_pos]
    # for dim_i_index, dim_pair_i in enumerate(dim_pair):
    #     points_dim_i = cut_indicator_p[:, dim_pair_i]
    #     points = points_dim_i[ConvexHull(points_dim_i).vertices, :]
    #     perimeter_i = Polygon(points).length
    #     points_edge.append(points)
    #     periemeter_seq[dim_i_index] = perimeter_i
    # cv_points[start_pos] = points_edge
    # cv_perimeters[start_pos] = periemeter_seq

    # print(start_pos)
    start_tree_pos = np.where(np.asarray(treeStr)[:, 0] == start_pos)[0][0]

    cost = 2*expon.rvs()/np.sum(cv_perimeters[start_pos])

    if ((cut_pos[start_pos][1] + cost) < tau_i):

        dim_index = np.random.choice(len(dim_pair), p=cv_perimeters[start_pos]/np.sum(cv_perimeters[start_pos]))
        sequentialPoints = cv_points[start_pos][dim_index]
        c_p = sequentialProcess_gen(sequentialPoints)


        cut_line_points[start_pos] = c_p
        cut_pos[start_pos][0] = (dim_index)
        cut_pos[start_pos][2] = (cost)

        treeStr[start_tree_pos][1] = len(cut_pos)
        treeStr[start_tree_pos][2] = len(cut_pos) + 1
        treeStr.append([len(cut_pos), -1, -1])
        treeStr.append([len(cut_pos) + 1, -1, -1])

        cu_index = np.where(np.asarray(label_per_data) == start_pos)[0]
        kx = current_data[cu_index][:, dim_pair[dim_index]]

        index_p = (((kx[:, 0] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (kx[:, 1] - c_p[0, 1]) * (
            c_p[1, 0] - c_p[0, 0])) < 0)

        label_seq.append(np.sum(index_p))
        label_seq.append(np.sum(~index_p))

        label_per_data[cu_index[index_p]] = treeStr[start_tree_pos][1]
        [cv_points, cv_perimeters] = update_cv(current_data, label_per_data, cu_index, index_p, dim_pair, treeStr[start_tree_pos][1], cv_points, cv_perimeters)
        label_per_data[cu_index[~index_p]] = treeStr[start_tree_pos][2]
        [cv_points, cv_perimeters] = update_cv(current_data, label_per_data, cu_index, index_p, dim_pair, treeStr[start_tree_pos][1], cv_points, cv_perimeters)

        cut_line_points.append([])
        cut_line_points.append([])
        cut_pos.append([np.nan, cut_pos[start_pos][1] + cost, np.nan])
        cut_pos.append([np.nan, cut_pos[start_pos][1] + cost, np.nan])

        [label_seq, label_per_data, cut_pos, cut_line_points, treeStr, cv_points, cv_perimeters] = CutConvexHull(label_seq,
                                        label_per_data, treeStr[start_tree_pos][1], dim_pair, current_data, cut_pos, tau_i, cut_line_points,
                                        treeStr, cv_points, cv_perimeters)
        [label_seq, label_per_data, cut_pos, cut_line_points, treeStr, cv_points, cv_perimeters] = CutConvexHull(label_seq,
                                        label_per_data, treeStr[start_tree_pos][2], dim_pair, current_data, cut_pos, tau_i, cut_line_points,
                                        treeStr, cv_points, cv_perimeters)
        #
        # index_p = (((xdata_i[dim_pair[dim_index][0]] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (
        # xdata_i[dim_pair[dim_index][1]] - c_p[0, 1]) * (c_p[1, 0] - c_p[0, 0])) < 0)
        # if index_p:
        #     start_pos = treeStr[start_pos][1]
        # else:
        #     start_pos = treeStr[start_pos][2]


    return label_seq, label_per_data, cut_pos, cut_line_points, treeStr, cv_points, cv_perimeters
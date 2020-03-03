import numpy as np
import scipy.io

import os
import pandas as pd

from scipy.stats import invgamma, uniform, expon
import scipy.stats as stats
from scipy.spatial import ConvexHull
import sys
import copy
from utility import *

class bspf_str:
    def __init__(self, minnum, dimNum):
        self.minnum = minnum
        self.dimNum = dimNum
        self.treeStr = [[0, -1, -1]]
        self.cut_pos = [[np.nan, 0, np.nan]] # 0: cut dimensional pair; 1, cut location in the budget line; 2, cost
        self.cut_line_points = [[]]

        dim_pair = []
        for k1 in range(dimNum-1):
            for k2 in np.arange(k1+1, dimNum):
                dim_pair.append([k1, k2])
        self.dim_pair = dim_pair

        self.cv_points = [[]]
        self.cv_perimeters = [[]]




    def update_new(self, xdata_i, current_data, label_per_data, label_seq, tau_i):
        cut_flag = True
        start_pos = self.treeStr[0][0]
        while cut_flag:
            # print(start_pos)
            cut_indicator_p = current_data[label_per_data == start_pos]
            if (label_seq[start_pos] < 3)|(len(np.unique(cut_indicator_p[:, 0]))<3):  # ensure the node has at least 3 data points
                label_seq[start_pos] += 1
                label_per_data = np.append(label_per_data, start_pos)
                return label_per_data, label_seq
            elif len(self.cv_points[start_pos])==0:

                for dim_pair_i in (self.dim_pair):
                    points_dim_i = cut_indicator_p[:, dim_pair_i]
                    cv_str = ConvexHull(points_dim_i)
                    self.cv_points[start_pos].append(points_dim_i[cv_str.vertices, :])
                    self.cv_perimeters[start_pos].append(perimeter_cal(self.cv_points[start_pos][-1]))

            start_tree_pos = np.where(np.asarray(self.treeStr)[:, 0]==start_pos)[0][0]
            # Include xdata_i in the current convex hull
            cv_points_new = []
            cv_perimeters_new = []
            for dimi in range(len(self.dim_pair)):
                new_dimi_points = np.vstack((self.cv_points[start_pos][dimi], xdata_i[self.dim_pair[dimi]][np.newaxis, :]))
                cv_str_new = ConvexHull(new_dimi_points)
                cv_points_new.append(new_dimi_points[cv_str_new.vertices])
                cv_perimeters_new.append(perimeter_cal(cv_points_new[-1]))  # points need to be index later


            # need to use the new formed convex hull now
            if self.treeStr[start_tree_pos][1]==(-1):
                new_current_data = np.vstack((current_data, xdata_i))
                label_seq[start_pos] += 1
                label_per_data = np.append(label_per_data, start_pos)
                self.cv_points[start_pos] = cv_points_new
                self.cv_perimeters[start_pos] = cv_perimeters_new
                [label_seq, label_per_data, cut_pos, cut_line_points, treeStr, cv_points, cv_perimeters] = CutConvexHull(label_seq, label_per_data,
                            start_pos, self.dim_pair, new_current_data, self.cut_pos, tau_i, self.cut_line_points,
                            self.treeStr, self.cv_points, self.cv_perimeters)
                self.cut_pos = cut_pos
                self.cut_line_points = cut_line_points
                self.treeStr = treeStr
                self.cv_points = cv_points
                self.cv_perimeters = cv_perimeters

                return label_per_data, label_seq

            differs = np.sum(cv_perimeters_new)-np.sum(self.cv_perimeters[start_pos])
            # print('finally: '+str(differs))
            if differs < 0:
                print('difference of perimeter is wrong !!')
            cost_new = 2*expon.rvs()/differs

            # find start_pos's previous loc
            # if start_pos == 0:
            # indicatos1 = [(self.treeStr[i1][1]==start_pos) for i1 in range(len(self.treeStr))]
            # indicatos2 = [(self.treeStr[i1][2]==start_pos) for i1 in range(len(self.treeStr))]
            if cost_new < self.cut_pos[start_pos][2]:
                p_unnorm = np.asarray([cv_perimeters_new[ci] - self.cv_perimeters[start_pos][ci] for ci in range(len(self.dim_pair))])
                pair_select = np.random.choice(len(self.dim_pair), p = p_unnorm/np.sum(p_unnorm))
                c_p = sequentialProcess_gen_no_crossing(self.cv_points[start_pos][pair_select], cv_points_new[pair_select])
                cut_block_index = len(self.cut_pos)
                index_p = (((xdata_i[self.dim_pair[pair_select][0]] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (
                        xdata_i[self.dim_pair[pair_select][1]] - c_p[0, 1]) * (c_p[1, 0] - c_p[0, 0])) < 0)
                if index_p:
                    # new_treeStr = np.array([[cut_block_index, cut_block_index + 1, start_pos], [cut_block_index + 1, -1, -1]])
                    new_treeStr = [[cut_block_index, cut_block_index + 1, start_pos], [cut_block_index + 1, -1, -1]]
                else:
                    # new_treeStr = np.array([[cut_block_index, start_pos, cut_block_index + 1], [cut_block_index + 1, -1, -1]])
                    new_treeStr = [[cut_block_index, start_pos, cut_block_index + 1], [cut_block_index + 1, -1, -1]]

                if start_tree_pos == self.treeStr[0][0]:
                    # self.treeStr = np.vstack((new_treeStr, self.treeStr))
                    self.treeStr.insert(0, new_treeStr[1])
                    self.treeStr.insert(0, new_treeStr[0])
                else:
                    indicatos1 = [(self.treeStr[i1][1] == start_pos) for i1 in range(len(self.treeStr))]
                    indicatos2 = [(self.treeStr[i1][2] == start_pos) for i1 in range(len(self.treeStr))]

                    if np.sum(indicatos1) == 1:
                        self.treeStr[np.where(indicatos1)[0][0]][1] = cut_block_index
                    elif np.sum(indicatos2) == 1:
                        self.treeStr[np.where(indicatos2)[0][0]][2] = cut_block_index
                    self.treeStr.extend(new_treeStr)
                    # self.treeStr = np.vstack((self.treeStr, new_treeStr))

                # update cv_points
                cv_points_new = []
                cv_perimeters_new = []
                for dimi in range(len(self.dim_pair)):
                    new_dimi_points = np.vstack(
                        (self.cv_points[start_pos][dimi], xdata_i[self.dim_pair[dimi]][np.newaxis, :]))
                    cv_str_new = ConvexHull(new_dimi_points)
                    cv_points_new.append(new_dimi_points[cv_str_new.vertices])
                    cv_perimeters_new.append(perimeter_cal(cv_points_new[-1]))
                self.cv_points.append(cv_points_new)
                self.cv_points.append([])
                self.cv_perimeters.append(cv_perimeters_new)
                self.cv_perimeters.append([])
                # udpate cv_perimeters

                self.cut_pos.append([pair_select, copy.copy(self.cut_pos[start_pos][1]),cost_new])
                self.cut_pos.append([np.nan, copy.copy(self.cut_pos[start_pos][1]),np.nan])
                self.cut_pos[start_pos][1] = self.cut_pos[start_pos][1] + cost_new
                self.cut_line_points.append(c_p)
                self.cut_line_points.append([])
                label_seq.extend([label_seq[start_pos], 1])
                label_per_data = np.append(label_per_data, cut_block_index+1)

                return label_per_data, label_seq
            else:
                if differs > 0:
                    self.cv_points[start_pos] = cv_points_new
                    self.cv_perimeters[start_pos] = cv_perimeters_new
                if np.isnan(self.cut_pos[start_pos][0]):
                    a = 1
                dim_index = int(self.cut_pos[start_pos][0])
                if start_pos >= len(self.cut_line_points):
                    a = 1
                c_p = self.cut_line_points[start_pos]
                index_p = (((xdata_i[self.dim_pair[dim_index][0]] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (xdata_i[self.dim_pair[dim_index][1]] - c_p[0, 1]) * (c_p[1, 0] - c_p[0, 0])) < 0)
                if index_p:
                    start_pos = self.treeStr[start_tree_pos][1]
                else:
                    start_pos = self.treeStr[start_tree_pos][2]

        return label_per_data, label_seq






    def cal_label(self, xdata):
        ydata_label = []
        for xdata_i_index, xdata_i in enumerate(xdata):
            cut_flag = True
            start_pos = self.treeStr[0][0]

            while cut_flag:
                start_tree_pos = np.where(np.asarray(self.treeStr)[:, 0] == start_pos)[0][0]
                if self.treeStr[start_tree_pos][1] == -1:
                    break
                c_p = self.cut_line_points[start_pos]
                dim_i = int(self.cut_pos[start_pos][0])
                data_dim_i = xdata_i[self.dim_pair[dim_i]]
                index_p = (((data_dim_i[0] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (data_dim_i[1] - c_p[0, 1]) * (c_p[1, 0] - c_p[0, 0])) < 0)
                if index_p:
                    start_pos = self.treeStr[start_tree_pos][1]
                else:
                    start_pos = self.treeStr[start_tree_pos][2]
                # print(start_pos)
            ydata_label.append(start_pos)
            if np.mod(xdata_i_index, 100)==0:
                print(str(xdata_i_index)+' has finished.')
        return ydata_label

    def sumDim_cal(self):
        candidate = np.where(np.asarray(self.treeStr)[:, 1]==-1)[0].astype(int)
        lower = [self.box_pos_lower[can_i] for can_i in candidate]
        upper = [self.box_pos_upper[can_i] for can_i in candidate]

        val = [np.sum(abs(upper[ii]-lower[ii])) for ii in range(len(lower))]
        return val



    def update(self, xdata_i, current_data, label_per_data, label_seq, tau_i):
        cut_flag = True
        start_pos = 0
        while cut_flag:
            if self.treeStr[start_pos][1]==0:
                break
            dim_index = self.cut_pos[start_pos][0]
            c_p = self.cut_line_points[start_pos]
            index_p = (((xdata_i[self.dim_pair[dim_index][0]] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (xdata_i[self.dim_pair[dim_index][1]] - c_p[0, 1]) * (c_p[1, 0] - c_p[0, 0])) < 0)
            if index_p:
                start_pos = self.treeStr[start_pos][1]
            else:
                start_pos = self.treeStr[start_pos][2]



        while (cut_flag):
            # if np.isnan(self.cut_pos[0]):

            # ensure the node has at least 3 data points
            if label_seq[start_pos] < 3:
                label_seq[start_pos] += 1
                label_per_data = np.append(label_per_data, start_pos)
                return label_per_data, label_seq

            points_edge = []
            periemeter_seq = np.zeros(len(self.dim_pair))
            cut_indicator_p = current_data[label_per_data == start_pos]
            for dim_i_index, dim_pair_i in enumerate(self.dim_pair):
                points_dim_i = cut_indicator_p[:, dim_pair_i]
                points = points_dim_i[ConvexHull(points_dim_i, incremental = True).vertices, :]
                perimeter_i = perimeter_cal(points)
                points_edge.append(points)
                periemeter_seq[dim_i_index] = perimeter_i

            cost = expon.rvs()/np.sum(periemeter_seq)

            if ((self.cut_pos[start_pos][1]+cost)<tau_i):

                dim_index = np.random.choice(len(self.dim_pair), p = periemeter_seq/np.sum(periemeter_seq))

                sequentialPoints = points_edge[dim_index]
                c_p = sequentialProcess_gen(sequentialPoints)

                self.cut_line_points.append(c_p)
                self.cut_pos[start_pos][0] = (dim_index)


                self.treeStr[start_pos][1] = len(self.cut_pos)
                self.treeStr[start_pos][2] = len(self.cut_pos)+1

                self.treeStr.append([len(self.cut_pos), 0, 0])
                self.treeStr.append([len(self.cut_pos)+1, 0, 0])

                cu_index = np.where(np.asarray(label_per_data)==start_pos)[0]
                kx = current_data[cu_index][:, self.dim_pair[dim_index]]


                index_p = (((kx[:, 0] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (kx[:, 1] - c_p[0, 1]) * (
                c_p[1, 0] - c_p[0, 0])) < 0)

                label_seq.append(np.sum(index_p))
                label_seq.append(np.sum(~index_p))

                label_per_data[cu_index[index_p]] = len(self.cut_pos)
                label_per_data[cu_index[~index_p]] = len(self.cut_pos)+1


                self.cut_pos.append([np.nan, self.cut_pos[start_pos][1]+cost])
                self.cut_pos.append([np.nan, self.cut_pos[start_pos][1]+cost])


                index_p = (((xdata_i[self.dim_pair[dim_index][0]] - c_p[0, 0]) * (c_p[1, 1] - c_p[0, 1]) - (xdata_i[self.dim_pair[dim_index][1]] - c_p[0, 1]) * (c_p[1, 0] - c_p[0, 0])) < 0)
                if index_p:
                    start_pos = self.treeStr[start_pos][1]
                else:
                    start_pos = self.treeStr[start_pos][2]
            else:
                label_seq[start_pos] += 1
                label_per_data = np.append(label_per_data, (start_pos))
                break
        return label_per_data, label_seq

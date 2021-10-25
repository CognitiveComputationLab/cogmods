import numpy as np
import os
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import csv
from rawdata.z_transform import z_transformation
import math


def get_raw(filename, shrink, z_transform):
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))


        for d_ix in range(len(data)):

            data[d_ix] = data[d_ix][0].split("\t")

        data = np.array(data, dtype = 'object')
        data = np.delete(data, 0, axis=0)

        for outer_ix in range(len(data)):
            for inner_ix in range(len(data[outer_ix])):
                try:
                    data[outer_ix, inner_ix] = int(data[outer_ix, inner_ix])
                except:
                    pass


        # remove vp data if there is a _b in name because i dont know what this means
        if False:
            removables = []
            for array_ix, vp in enumerate(data[:,2]):
                if "_b" in vp:
                    removables.append(array_ix)
            data = np.delete(data, removables, axis=0)


        # get only TRUE hits for taliahich data


        premises = np.unique(data[:, 10])


        com_times = []
        mss = []
        for p_ix, p in enumerate(premises):
            p_times = np.unique(data[np.where(data[:, 10] == p), 11])
            for pt_ix, pt in enumerate(p_times):
                com_times.append(p_ix * (p_times[-1] + p_times[1]) + pt)
                this = data[np.where(np.logical_and(data[:, 10] == p, data[:, 11] == pt)), 8]
                mss.append(this)




    # outlier correction
    """
    cut_mss = []
    for ms in mss:
        localmean = np.mean(ms)
        localstd = np.std(ms)

        temp = []

        for m in ms[0]:
            if (m < (localmean + 2 * localstd)) and (m > (localmean - 2 * localstd)):
                temp.append(m)


        cut_mss.append(np.array([temp],  dtype = 'object'))


    mss = cut_mss
    """
    means = []
    stds = []
    errors = []

    if shrink == 1:
        for ms in mss:
            means.append(np.mean(ms))
            stds.append(np.std(ms))

        for ms in mss:
                errors.append(np.std(ms) / math.sqrt(len(ms)) )

    else:
        means = []
        stds = []
        for i in range(int(math.ceil(len(mss)/shrink))):
            temp = []
            for j in range(shrink):
                if i*shrink+j < len(mss):
                    temp.append(mss[i*shrink+j])

            if len(temp) > 1:
                temp = np.concatenate(temp, axis=1)
            else:
                pass


            means.append(np.mean(temp))


            errors.append(np.std(temp) / math.sqrt(len(temp[0])) )
            # input(errors)








    if z_transform:




        plusmeans = np.array(means) + np.array(errors)
        minusmeans = np.array(means) - np.array(errors)



        overall_mean = np.mean(means)
        std = np.std(means)
        for m_ix, m in enumerate(means):
            means[m_ix] = (means[m_ix] - overall_mean) / std

        for m_ix, m in enumerate(plusmeans):
            plusmeans[m_ix] = (plusmeans[m_ix] - overall_mean) / std

        for m_ix, m in enumerate(minusmeans):
            minusmeans[m_ix] = (minusmeans[m_ix] - overall_mean) / std



        errors = (plusmeans - minusmeans) / 2


        return np.array(means), np.array(errors)







    return np.array(means), np.array(errors)








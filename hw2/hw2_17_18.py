import numpy as np
from numpy import random
import sys

def get_sign(d):                    #符号函数，根据数值正负返回1，-1
    sign = np.ones(d.shape)
    for i,val in enumerate(d):
        if val <= 0:
            sign[i] = -1
    return sign


def Generate_data(low=-1,high=1,size=20,noise_rate=0.2):
    data = random.uniform(low,high,size)
    data_sign = get_sign(data)
    temp = random.uniform(size=size) - noise_rate
    noise = get_sign(temp)
    sign = data_sign*noise
    if len(data) != len(sign):
        sys.exit(-1)
    return data,sign

def get_theta(data,theta_num):
    theta = np.zeros(theta_num)
    data_lh = sorted(data)
    for i in range(theta_num-1):
        theta[i] = (data_lh[i] +data_lh[i+1])*0.5
    theta[-1] = data_lh[-1]+1
    return theta

def get_err(data,sign,theta,s):
    data_size = len(data)
    err_count = 0
    res = sign*(s*get_sign(data-theta))
    err_count = (data_size - np.sum(res))/2
    return err_count/data_size

def Dec_stump(data,sign):
    theta_num = len(data)
    theta = get_theta(data,theta_num)
    err_array = np.zeros((2,theta_num))
    for i in range(theta_num):
        err_array[0][i] = get_err(data,sign,theta[i],s=1)
        err_array[1][i] = get_err(data,sign,theta[i],s=-1)
    min_0 = np.min(err_array[0])
    min_1 = np.min(err_array[1])

    if min_0 < min_1:
        err_min = min_0
        s_best = 1
        theta_best = theta[np.argmin(err_array[0])]
    else:
        err_min = min_1
        s_best = -1
        theta_best = theta[np.argmin(err_array[1])]

    return s_best,theta_best,err_min

if __name__ == '__main__':
    total = 1000
    err_train_sum = 0
    err_expt_sum = 0
    for i in range(total):
        data_size = 20
        data,sign = Generate_data(low=-1,high=1,size=data_size,noise_rate=0.2)
        s,theta,err_train = Dec_stump(data,sign)
        err_expt = 0.5+0.3*s*(np.abs(theta)-1)

        err_train_sum += err_train
        err_expt_sum += err_expt
        err_train_list = []
        err_expt_list = []
        err_train_list.append(err_train)
        err_expt_list.append(err_expt)
    print('average of E_in:%.3f\naverage of E_out:%.3f'
            %(err_train_sum/total,err_expt_sum/total))
# average of E_in:0.168
# average of E_out:0.261

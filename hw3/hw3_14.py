import numpy as np
from numpy import random as ran
import sys

def get_sign(d):                    #符号函数，根据数值正负返回1，-1
    sign = np.ones(d.shape)
    for i,val in enumerate(d):
        if val <= 0:
            sign[i] = -1
    return sign

def Generate_data(low=-1,high=1,size=1000,noise_rate=0.1):
    data = np.ones((size,6))
    x1 = ran.uniform(low,high,size)
    x2 = ran.uniform(low,high,size)
    x1_x2 = np.multiply(x1,x2)
    x1_x1 = np.multiply(x1,x1)
    x2_x2 = np.multiply(x2,x2)
    for i in range(size):
        data[i][1] = x1[i]
        data[i][2] = x2[i]
        data[i][3] = x1_x2[i]
        data[i][4] = x1_x1[i]
        data[i][5] = x2_x2[i]

    res_f = np.zeros(size)
    for i in range(size):
        res_f[i] = data[i][1]*data[i][1] + data[i][2]*data[i][2] - 0.6
    sign_f = get_sign(res_f)

    temp = ran.uniform(size=size) - noise_rate
    noise = get_sign(temp)
    sign = np.multiply(sign_f,noise)
    if data.shape[0] != sign.shape[0]:
        sys.exit(-1)
    return data,sign

def Lin_Reg(x,y):
    x_pinv = np.linalg.pinv(x)
    w = np.dot(x_pinv,y)
    return w

def get_err(x,y,w):
    size = x.shape[0]
    y_w = np.dot(x,w)
    y_w = get_sign(y_w)

    temp = np.multiply(y,y_w)
    err_count = (size - np.sum(temp))/2
    return err_count/size

if __name__ == '__main__':
    run_times = 1000
    size = 1000
    noise_rate = 0.1
    err_list = []
    w3_list = []

    for i in range(run_times):
        data,sign = Generate_data(size = 1000,noise_rate = 0.1)
        w_lin = Lin_Reg(x=data,y=sign)
        err_train = get_err(x=data,y=sign,w = w_lin)

        w3_list.append(w_lin[3])
        err_list.append(err_train)

        # if i<5:
        #     print(i,w_lin)

    sum = 0
    for i in w3_list:
        sum = sum + i
    print(sum/run_times)            # 0.00287693932433

    total = 0
    for i in err_list:
        total = total + i
    print(total/run_times)          # 0.124661



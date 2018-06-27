import numpy as np
import sys


def Transpose(data,sign):
    data_tr = np.zeros((len(data[0]), len(data)))
    sign_tr = np.zeros(len(sign))
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_tr[j][i] = data[i][j]
        sign_tr[i] = sign[i]
    return data_tr, sign_tr


def Readfile(file_name):
    data = []
    sign = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            items = line.split()
            for i in range(len(items)):         #转换成数值类型
                items[i] = float(items[i])
            data.append(items[0:-1])            #加入data
            sign.append(items[-1])              #加入sign
    if (len(data) != len(sign)):
        sys.exit(-1)
    data_tr,sign_tr = Transpose(data,sign)
    return data_tr,sign_tr


def get_sign(d):                    #符号函数，根据数值正负返回1，-1
    sign = np.ones(d.shape)
    for i,val in enumerate(d):
        if val <= 0:
            sign[i] = -1
    return sign


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
    file_train = 'hw2_train.dat'
    file_test = 'hw2_test.dat'
    data,sign = Readfile(file_train)
    data_test,sign_test =Readfile(file_test)
    if len(data) != len(data_test):
        sys.exit(-1)

    dimension = len(data)
    s_array = np.zeros(dimension)
    theta_array = np.zeros(dimension)
    err_array = np.zeros(dimension)
    for i in range(dimension):
        s,theta,err_train = Dec_stump(data[i],sign)
        s_array[i] = s
        theta_array[i] = theta
        err_array[i] = err_train

    err_min = np.min(err_array)
    dim_best = np.argmin(err_array)
    s_best,theta_best = s_array[dim_best], theta_array[dim_best]

    err_test = get_err(data_test[dim_best],sign_test,theta_best,s_best)

    print(s_best,theta_best,dim_best)       # -1 1.6175 3
    print(err_min,err_test)                 # 0.25 0.355




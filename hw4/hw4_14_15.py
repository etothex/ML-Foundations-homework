import numpy as np
import sys

def get_data(file):
    data = []
    sign = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            words = line.split()
            for i in range(len(words)):
                words[i] = float(words[i])
            words.insert(0,1.0)             #增加一个维度！
            data.append(words[0:-1])
            sign.append(words[-1])
    data = np.array(data)
    sign = np.array(sign)
    if data.shape[0] != sign.shape[0]:
        sys.exit(-1)
    return data,sign

def Lin_Reg_reg(x,y,m):
    r1 = np.dot(x.transpose(),x)
    r2 = np.dot(np.eye(x.shape[1]),m)
    r3 = r1+r2
    r4 = np.linalg.inv(r3)
    r5 = np.dot(x.transpose(),y)
    w = np.dot(r4,r5)

    return w

def get_sign(d):                # 符号函数：数值为正为+1，否则，为-1
    sign = np.ones(d.shape)
    for i, val in enumerate(d):
        if val <= 0:
            sign[i] = -1
    return sign

def get_err(x,y,w):
    size = x.shape[0]
    y_w = np.dot(x,w)
    y_w = get_sign(y_w)

    temp = np.multiply(y,y_w)
    err_count = (size - np.sum(temp))/2
    return err_count/size

if __name__ == "__main__":
    file_train = 'hw4_train.dat'
    file_test = 'hw4_test.dat'
    data,sign = get_data(file_train)
    data_test,sign_test = get_data(file_test)

    w_list = []
    err_train_list = []
    err_test_list = []
    multiplier = np.linspace(2,-10,13)

    for i in multiplier:
        m = np.power(10,i)
        w = Lin_Reg_reg(x=data, y=sign, m=m)
        w_list.append(w)
        err_train_list.append(get_err(x=data, y=sign, w=w))
        err_test_list.append(get_err(x=data_test,y=sign_test,w = w))

    multiplier = np.array(multiplier)
    err_train_list = np.array(err_train_list)
    err_test_list = np.array(err_test_list)

    min_err_train_index = np.argmin(err_train_list)
    print(multiplier[min_err_train_index])
    print(err_train_list[min_err_train_index])
    print(err_test_list[min_err_train_index])
    '''
    -8.0
    0.015
    0.02
    '''
    min_err_test_index = np.argmin(err_test_list)
    print(multiplier[min_err_test_index])
    print(err_train_list[min_err_test_index])
    print(err_test_list[min_err_test_index])
    '''
    -7.0
    0.03
    0.015
    '''
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


def get_err(x, y, w):
    size = x.shape[0]
    y_w = np.dot(x,w)
    y_w = get_sign(y_w)

    temp = np.multiply(y,y_w)
    err_count = (size - np.sum(temp))/2
    ret = err_count / size
    return ret


def split(data, sign, val_part, folds):
    total = data.shape[0]
    split_size = int(total/folds)
    if (total%folds) != 0:
        print('Please choose folds again')
        sys.exit(-1)

    data_split = []
    sign_split = []
    for i in range(folds):
        data_split.append(np.array(data[i*split_size:(i+1)*split_size]))
        sign_split.append(np.array(sign[i*split_size:(i+1)*split_size]))

    data_val = data_split.pop(val_part)
    sign_val = sign_split.pop(val_part)

    data_train = data_split[0]
    sign_train = sign_split[0]
    for i in range(1,folds-1):
        data_train = np.vstack((data_train,data_split[i]))
        sign_train = np.hstack((sign_train,sign_split[i]))

    return data_train,sign_train,data_val,sign_val


if __name__ == "__main__":
    file_train = 'hw4_train.dat'
    file_test = 'hw4_test.dat'
    data,sign = get_data(file_train)
    data_test,sign_test = get_data(file_test)

    E_cv_list = []
    multiplier = np.linspace(2,-10,13)
    folds = 5

    for i in multiplier:
        m = np.power(10,i)
        err_val_list = np.ones(folds)
        for j in range(folds):
            data_train,sign_train,data_val,sign_val = split(data,sign,j,folds)
            w = Lin_Reg_reg(x=data_train, y=sign_train, m=m)
            err_val_list[j] = get_err(x=data_val, y=sign_val, w=w)
        E_cv = np.sum(err_val_list)/folds
        E_cv_list.append(E_cv)

    E_cv_list = np.array(E_cv_list)
    multiplier = np.array(multiplier)

    min_E_cv_index = np.argmin(E_cv_list)
    min_E_cv = E_cv_list[min_E_cv_index]
    log_opt_m = multiplier[min_E_cv_index]
    opt_m = np.power(10,log_opt_m)
    w = Lin_Reg_reg(x=data, y=sign, m=opt_m)
    E_in = get_err(x=data, y=sign, w=w)
    E_out = get_err(x=data_test, y=sign_test, w=w)
    print(log_opt_m)
    print(min_E_cv)
    print(E_in)
    print(E_out)
    '''
    -8.0
    0.03
    0.015
    0.02
    '''




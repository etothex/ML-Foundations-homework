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

    multiplier = 11.26

    w = Lin_Reg_reg(x=data,y=sign,m=multiplier)
    err_train = get_err(x=data, y=sign, w=w)
    err_test= get_err(x=data_test,y=sign_test,w = w)

    print(w)                        # [-0.88765371  1.00531461  1.00530077]
    print(err_train)                # 0.055
    print(err_test)                 # 0.052
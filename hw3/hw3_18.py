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

def Log_Reg(x,y,w0,step,T):
    size = x.shape[0]
    w = w0

    for i in range(T):
        nabla_err = np.zeros(x.shape[1])
        for i in range(size):
            val1 = np.dot(x[i],w)
            val2 = -1 * y[i] * val1
            val3 = 1 / (1 + np.exp(-1*val2))
            val = val3 * (-1) * y[i] * x[i]
            nabla_err = nabla_err + val
        nabla_Ein = nabla_err/size

        w = w - step*nabla_Ein

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
    file_train = 'hw3_train.dat'
    file_test = 'hw3_test.dat'
    data,sign = get_data(file_train)
    data_test,sign_test = get_data(file_test)

    w_init = np.zeros(data.shape[1])
    update_times = 2000
    # update_times = 20000
    step = 0.001

    w = Log_Reg(x=data,y=sign,w0=w_init,step=step,T=update_times)
    err_test= get_err(x=data_test,y=sign_test,w = w)

    print(w)
    print(err_test)

'''update_times = 2000,step = 0.001
[ 0.01878417 -0.01260595  0.04084862 -0.03266317  0.01502334 -0.03667437
  0.01255934  0.04815065 -0.02206419  0.02479605  0.06899284  0.0193719
 -0.01988549 -0.0087049   0.04605863  0.05793382  0.061218   -0.04720391
  0.06070375 -0.01610907 -0.03484607]
0.475
'''
'''update_times = 20,000,step = 0.001
[-0.0038536  -0.18913185  0.26624136 -0.35353837  0.04088378 -0.379398
  0.01982524  0.33389516 -0.26384736  0.134886    0.49138154  0.08725686
 -0.25536037 -0.16290837  0.30071379  0.40011899  0.43215705 -0.46224099
  0.43227252 -0.20784597 -0.36933394]
0.22
'''
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
            words.insert(0, 1.0)             #增加一个维度！
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
    # update_times = 10000
    step = 0.01

    w = Log_Reg(x = data,y = sign,w0 = w_init,step=step,T=update_times)
    err_test= get_err(x = data_test,y = sign_test,w = w)

    print(w)
    print(err_test)

'''update_times = 2000,step = 0.01
[-0.00385379 -0.18914564  0.26625908 -0.35356593  0.04088776 -0.3794296
  0.01982783  0.33391527 -0.26386754  0.13489328  0.4914191   0.08726107
 -0.25537728 -0.16291797  0.30073678  0.40014954  0.43218808 -0.46227968
  0.43230193 -0.20786372 -0.36936337]
0.22
'''
'''update_times = 10000,step = 0.01
[-0.09813936 -0.57957473  0.77179428 -1.02114211  0.05737681 -1.04653388
 -0.00634968  1.02637028 -0.76429381  0.40014953  1.3357829   0.25668254
 -0.81391661 -0.550549    0.80468791  1.08266316  1.21775141 -1.26616194
  1.25030156 -0.5793083  -1.0344671 ]
0.183333333333
'''
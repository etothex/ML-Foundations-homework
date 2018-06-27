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
        # nabla_err = np.zeros(x.shape[1])
        # for i in range(size):
        #     val1 = np.dot(x[i],w)
        #     val2 = -1 * y[i] * val1
        #     val3 = 1 / (1 + np.exp(-1*val2))
        #     val = val3 * (-1) * y[i] * x[i]
        #     nabla_err = nabla_err + val
        # nabla_Ein = nabla_err/size
        n = i%size
        val1 = np.dot(x[n],w)
        val2 = -1 * y[n] * val1
        val3 = 1 / (1 + np.exp(-1*val2))
        nabla_err = val3 * (-1) * y[n] * x[n]

        w = w - step * nabla_err

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
    data, sign = get_data(file_train)
    data_test, sign_test = get_data(file_test)

    w_init = np.zeros(data.shape[1])
    update_times = 2000
    # update_times = 100000
    step = 0.001

    w = Log_Reg(x=data,y=sign,w0=w_init,step=step,T=update_times)
    err_test = get_err(x=data_test,y=sign_test,w=w)

    print(w)
    print(err_test)

'''update_times = 2000,step = 0.001
[ 0.01826899 -0.01308051  0.04072894 -0.03295698  0.01498363 -0.03691042
  0.01232819  0.04791334 -0.02244958  0.02470544  0.06878235  0.01897378
 -0.02032107 -0.00901469  0.04589259  0.05776824  0.06102487 -0.04756147
  0.06035018 -0.01660574 -0.03509342]
0.473
'''
''' update_times = 100,000,step = 0.001
[-0.09590889 -0.58232703  0.77476974 -1.02000126  0.06063709 -1.04590293
 -0.00580349  1.0278774  -0.76345124  0.40320762  1.33731943  0.25856417
 -0.81485107 -0.5514648   0.80557756  1.08442632  1.21786281 -1.26634801
  1.25095197 -0.58310454 -1.0315415 ]
0.186333333333
'''
'''update_times = 200,000,step = 0.001
[-0.19625712 -0.76422315  1.02772369 -1.32148645  0.05087293 -1.31915092
 -0.03677788  1.40868811 -0.97140656  0.53271546  1.69921267  0.34557508
 -1.10474347 -0.74962585  1.02283857  1.37822401  1.58027258 -1.60188215
  1.64050203 -0.73266664 -1.29331438]
0.181333333333
'''
'''update_times = 1,000,000,step = 0.001
[-0.57272507 -0.89319661  1.27998748 -1.56083697  0.06994605 -1.52385046
 -0.04933554  1.81354971 -1.11183264  0.66512494  2.02899289  0.4431238
 -1.36771983 -0.90197444  1.23139163  1.65199747  1.92946162 -1.86376779
  2.01096867 -0.817384   -1.47081153]
0.182666666667
'''
'''update_times = 1,000,000,step = 0.01
[-0.68552639 -0.93847846  1.2992892  -1.55489463  0.10502183 -1.52714896
 -0.04065551  1.84109572 -1.12224144  0.67559278  2.06961908  0.43902009
 -1.39588851 -0.8924943   1.25585593  1.67334801  1.93533871 -1.8790418
  2.01954127 -0.85342721 -1.47452177]
0.180333333333
'''
''' update_times = 10,000,000,step = 0.01
[-0.68552639 -0.93847845  1.2992892  -1.55489463  0.10502183 -1.52714896
 -0.04065551  1.84109572 -1.12224144  0.67559278  2.06961908  0.4390201
 -1.39588851 -0.8924943   1.25585593  1.67334801  1.93533871 -1.8790418
  2.01954127 -0.85342721 -1.47452177]
0.180333333333
'''
'''update_times = 200,000,step = 0.01
[-0.67117899 -0.9401475   1.29829842 -1.5565333   0.10307424 -1.52815992
 -0.04216656  1.8397434  -1.12357009  0.67370591  2.06800733  0.43758835
 -1.39656525 -0.89395155  1.25461922  1.67196564  1.93377925 -1.88039094
  2.01786815 -0.85507344 -1.47574561]
0.18
'''
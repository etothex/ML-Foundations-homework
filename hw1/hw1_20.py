import sys
import numpy as np
import random

def Readfile(filename):
    data = []
    sign = []
    with open(filename, 'r') as f:
        for line in f:
            line=line.strip()
            items = line.split()
            for i in range(len(items)):
                items[i] = float(items[i])
            items.insert(0,1.0)
            data.append(items[0:-1])
            sign.append(int(items[-1]))
    if len(data) != len(sign):
        sys.exit(-1)
    return data,sign

def error_weight(data,sign,w):
    error_num = 0
    total = len(data)
    for i in range(total):
        if len(w) != len(data[i]):
            sys.exit(-1)
        score = np.dot(w,data[i])
        tag = 1 if score > 0 else -1
        if tag != sign[i]:
            error_num += 1
    ret = error_num/total
    return ret

def find_error(data,sign,w):
    total = len(data)
    rd = random.Random()
    for i in range(total*10):
        index = rd.randint(0,total-1)
        if len(w) != len(data[index]):
            sys.exit(-1)
        score = np.dot(w, data[index])
        tag = 1 if score > 0 else -1
        if tag != sign[index]:
            return True,index
    else:
        err_train = error_weight(data,sign,w)
        if err_train != 0:
            while True:
                index = rd.randint(0, total - 1)
                if len(w) != len(data[index]):
                    sys.exit(-1)
                score = np.dot(w, data[index])
                tag = 1 if score > 0 else -1
                if tag != sign[index]:
                    return True, index
        else:
            index = total
            return False,index

def update_w(data,sign,error_index,w,k=1):

    temp = np.dot(sign[error_index],data[error_index])
    if (len(w) != len(temp)):
        sys.exit(-1)
    w = w + k*temp
    return w

def PLA_pocket(data,sign,w=list([0,0,0,0,0]),steps=50,k=1):
    w_best = w
    err_min = error_weight(data,sign,w_best)

    for i in range(steps):
        exit_error,err_index = find_error(data,sign,w)
        if exit_error:
            w = update_w(data,sign,err_index,w,k)
            err_train = error_weight(data,sign,w)
            if err_train < err_min:
                w_best = w
                err_min = err_train
        else:
            w_best = w
            err_min = 0
            break
    return w_best,err_min

if __name__ ==  '__main__':
    file_train = 'hw1_18_train.dat'
    data,sign = Readfile(file_train)

    file_test = 'hw1_18_test.dat'
    data_test,sign_test = Readfile(file_test)

    err_list=[]
    for i in range(100):
        w = [0,0,0,0,0]
        w, err_percent = PLA_pocket(data,sign,w,100)

        err_test = error_weight(data,sign,w)
        err_list.append(err_test)
        print(err_test,end=' ')
        if((i+1)%20 == 0):
            print()
    total = 0
    for i in err_list:
        total = total+i
    print('average of test error:%.3f' %(total/100))


# average of test error:0.107
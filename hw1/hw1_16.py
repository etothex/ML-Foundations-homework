import numpy as np
import sys
import random

def Readfile(filename):
    '''
    读取文件，前几列是数据，最后一列是标记：1、-1
    :param filename:文件名
    :return:data：数据列表,sign：标记列表
    '''
    data = []
    sign = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            items = line.split()
            for i in range(len(items)):         #转换成数值类型
                items[i] = float(items[i])
            items.insert(0,1.0)                 #增加一个维度!!!
            data.append(items[0:-1])            #加入data
            sign.append(items[-1])              #加入sign
    if len(data) != len(sign):
        sys.exit(-1)
    return data,sign

def find_mistake(data,sign,w,index_random,pt):
    '''
    index_random[pt] is index of last 'mistake' data in data lsit,
    so it is the begining of next cycle.
    '''

    pt_1 = pt
    right_num = 0                               #number of 'right' data
    total = len(sign)                           #total number of data
    index = index_random[pt_1]

    while right_num < total:                    # cycle,find 'mistake' data
        if (len(w) != len(data[index])):        # gurantee the same length
            sys.exit(-1)
        score = np.dot(w, data[index])

        # print('{0}\t\t{1:.3f}\t'.format(index+1,score),end='')
        # print('[',end='')
        # for i in w:
        #     print('%.3f'%i,end=' ')
        # print(']')

        if (score * sign[index] > 0) or (score == 0 and sign[index] < 0):
            right_num += 1
            pt_1 += 1
            if pt_1 == len(index_random):
                pt_1 = 0
            index = index_random[pt_1]
        else:
            pt = pt_1
            return False,pt
    else:
        return True,pt

def update_w(w,data,sign,halt_last,k=1):
    '''
    update w
    :param w: 
    :param data: 
    :param sign: 
    :param halt_last:错误数据的索引
    :param k: 
    :return: w
    '''
    temp = np.dot(sign[halt_last],data[halt_last])
    if (len(w) != len(temp)):                   # gurantee the same length
        sys.exit(-1)
    w = w + k*temp
    return w

def PLA(data,sign,w=list([0,0,0,0,0]),k=1):
    '''
    实现PLA算法。
    :param data: 数据
    :param sign: 标记
    :param w: 初始权重
    :param k: 调节w更改的大小
    :return: w：权重, steps：更新次数, halt_last：最后一次‘出错’数据的索引
    '''
    #生成随机序列，index_random是将data list的索引打乱后的序列，pt是index_random的索引
    index_random=[]
    for i in range(len(sign)):
        index_random.append(i)
    random.shuffle(index_random)
    pt = 0
    #更新步数
    steps = 0

    while(True):
        no_err, pt= find_mistake(data,sign,w,index_random,pt)     #find next mistake
        if not no_err:
            steps += 1
            halt_last = index_random[pt]
            w = update_w(w,data,sign,halt_last,k)                   #update w
        else:
            halt_last = index_random[pt]
            return w, steps, halt_last


if __name__ == '__main__':
    file_name = 'hw1_15_train.dat'
    data, sign = Readfile(file_name)
    # test Readfile
    # for i in range(len(sign)):
    #     print(data[i],sign[i])
    update_list = []

    for i in range(200):
        w = [0,0,0,0,0]
        w,steps,halt_last = PLA(data,sign,w)

        update_list.append(steps)
        print(steps,end=' ')
        if((i+1)%40 == 0):
            print()

        # print('w:','[', end='')
        # for i in w:
        #     print('%.3f' % i, end=' ')
        # print(']')
        # print('update:%s' %steps)
        # print('last_modification:\nline:%s' %(halt_last+1),end=' ')
        # print('data:',end='')
        # for i in data[halt_last][1:]:
        #     print('%s'%i,end=' ')
        # print('%d'%sign[halt_last])

    total = 0
    for i in update_list:
        total =total + i
    print('The average step is',total/len(update_list))




import math
from scipy.optimize import fsolve

#hw2_3
def vc_bound(epsilon,n,dvc):
    val = 1-(4*((2*n)**dvc)*math.exp((-1/8)*(epsilon**2)*n))
    return val

def hw2_3():
    epsilon = 0.05
    dvc = 10
    n = 100
    while True:
        val = vc_bound(epsilon,n,dvc)
        if val >= 0.95:
            print(n,val)
            break
        else:
            n = n + 100
#hw2_4
def func_p(x,n =10000,dvc = 50,deta = 0.05):
    e = x[0]

    return[
        math.sqrt((1/n)*(2*e+(math.log(6*(2*n)**dvc*(1/deta))))) - e
    ]

def func_d(x,n =10000,dvc = 50,deta = 0.05):
    e = x[0]

    return [
        math.sqrt((1/(2*n))*((4*e + 4*e*e) + math.log(4/deta) + dvc*math.log(n*n))) - e
    ]

def func_ovc(n,deta,dvc):
    e = math.sqrt(8/n*math.log(4*((2*n)**dvc)/deta))
    return e

def func_vvc(n, deta, dvc):
    e = math.sqrt(16/n*math.log(2*(n**dvc)/math.sqrt(deta)))
    return e

def func_rpb(n, deta, dvc):
    e = math.sqrt(2*math.log(2*n*(n**dvc))/n) + math.sqrt(2/n*math.log(1/deta)) + 1/n
    return e

def hw2_4():
    result = []
    n = 10000
    deta = 0.05
    dvc = 50
    result.append(func_ovc(n,deta,dvc))
    result.append(func_vvc(n, deta, dvc))
    result.append(func_rpb(n, deta, dvc))
    x1 = [0]
    x2 = [0]
    result.append(fsolve(func_p,x1)[0])
    result.append(fsolve(func_d,x2)[0])
    print(result)

def hw2_5():
    result = []
    n = 5
    deta = 0.05
    dvc = 50
    result.append(func_ovc(n ,deta, dvc))
    result.append(func_vvc(n, deta, dvc))
    result.append(func_rpb(n, deta, dvc))
    x1 = [0]
    x2 = [0]
    result.append(fsolve(func_p,x1,(5))[0])
    result.append(fsolve(func_d,x2,(5,50,0.05))[0])
    print(result)
if __name__ == "__main__":
    hw2_3()                             # 453000 0.9506224686204758
    hw2_4()                             # [0.632, 0.860, 0.331, 0.224, 0.215]
    hw2_5()                             # [13.828, 16.264, 7.048, 5.101, 5.593]

import numpy as np
# 将一个10进制正数转换为一个2进制数，保留m位整数，f位小数，首位符号位
def p_d2b(n, m, f):
    b = []
    x = 2
    n = n * np.power(2, f)
    n = int(n)
    while True:
        s = n // x
        y = n % x
        b = b + [y]
        if s == 0:
            break
        n = s
    b.reverse()
    if(len(b) > (m+f)):
        for i in range(m+f):
            b[i] = 1
            b = b[:m+f]
    elif(len(b) < (m+f)):
        for i in range(m+f-len(b)):
            b.insert(0,0)
    b.insert(0,0)
    a = [str(i) for i in b ]
    return a
#求一个10进制负数转换为一个2进制补码形式，保留m位整数，f位小数，首位符号位
def n_d2b(n, m, f):
    n = -1 * n
    b = p_d2b(n, m, f)
    b[0] = '1'
    flag = 1
    for i in range(len(b)-1,0,-1):
        if b[i]== '1' and flag == 1:
            b[i] = '1'
            flag = 0
        elif b[i] == '0' and flag == 1:
            b[i] = '0'
            flag = 1
        elif b[i] == '0':
            b[i] = '1'
        else:
            b[i] = '0'
    a = [str(i) for i in b ]
    return a
# 求一个数n的补码，保留m位整数，n位小数，首位符号位
def d2b(n, m, f):
    if n < 0:
        c = n_d2b(n, m, f)
    else:
        c = p_d2b(n, m, f)
    return c
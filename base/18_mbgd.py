# -*- coding: utf-8 -*-
'''
MBGD（Mini-batch gradient descent）小批量梯度下降：每次迭代使用b组样本
from:http://blog.csdn.net/xiaoch1222/article/details/52847521
'''
import random
#用y = Θ1*x1 + Θ2*x2来拟合下面的输入和输出
#input1  1   2   5   4
#input2  4   5   1   2
#output  19  26  19  20
input_x = [[1,4], [2,5], [5,1], [4,2]]  #输入
y = [19,26,19,20]       #输出
theta = [1,1]           #θ参数初始化
loss = 10               #loss先定义一个数，为了进入循环迭代
step_size = 0.01        #步长
eps =0.0001             #精度要求
max_iters = 10000       #最大迭代次数
error =0                #损失值
iter_count = 0          #当前迭代次数


while( loss > eps and iter_count < max_iters):  #迭代条件
    loss = 0
    #这里每次批量选取的是2组样本进行更新，另一个点是随机点+1的相邻点
    i = random.randint(0,3)     #随机抽取一组样本
    j = (i+1)%4                 #抽取另一组样本，j=i+1
    pred_y0 = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]  #预测值1
    pred_y1 = theta[0]*input_x[j][0]+theta[1]*input_x[j][1]  #预测值2
    theta[0] = theta[0] - step_size * (1/2) * ((pred_y0 - y[i]) * input_x[i][0]+(pred_y1 - y[j]) * input_x[j][0])  #对应5式
    theta[1] = theta[1] - step_size * (1/2) * ((pred_y0 - y[i]) * input_x[i][1]+(pred_y1 - y[j]) * input_x[j][1])  #对应5式
    for i in range (4):
        pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]     #总预测值
        error = (1/(2*2))*(pred_y - y[i])**2                    #损失值
        loss = loss + error       #总损失值
    iter_count += 1
    print ('iters_count', iter_count)

print ('theta: ',theta )
print ('final loss: ', loss)
print ('iters: ', iter_count)

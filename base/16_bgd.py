# -*- coding: utf-8 -*-
'''
  BGD(Batch gradient descent)批量梯度下降法：每次迭代使用所有的样本
  from:http://blog.csdn.net/xiaoch1222/article/details/52847521
'''
import random
#用y = Θ1*x1 + Θ2*x2来拟合下面的输入和输出
#input1  1   2   5   4
#input2  4   5   1   2
#output  19  26  19  20
input_x = [[1,4], [2,5], [5,1], [4,2]]  #输入
y = [19,26,19,20]   #输出
theta = [1,1]       #θ参数初始化
loss = 10           #loss先定义一个数，为了进入循环迭代
step_size = 0.01    #步长
eps =0.0001         #精度要求
max_iters = 10000   #最大迭代次数
error =0            #损失值
iter_count = 0      #当前迭代次数

err1=[0,0,0,0]      #求Θ1梯度的中间变量1
err2=[0,0,0,0]      #求Θ2梯度的中间变量2

while( loss > eps and iter_count < max_iters):   #迭代条件
    loss = 0
    err1sum = 0
    err2sum = 0
    for i in range (4):     #每次迭代所有的样本都进行训练
        pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]  #预测值
        err1[i]=(pred_y-y[i])*input_x[i][0]
        err1sum=err1sum+err1[i]
        err2[i]=(pred_y-y[i])*input_x[i][1]
        err2sum=err2sum+err2[i]
    theta[0] = theta[0] - step_size * err1sum/4  #对应5式
    theta[1] = theta[1] - step_size * err2sum/4  #对应5式
    for i in range (4):
        pred_y = theta[0]*input_x[i][0]+theta[1]*input_x[i][1]   #预测值
        error = (1/(2*4))*(pred_y - y[i])**2  #损失值
        loss = loss + error  #总损失值
    iter_count += 1
    print ("iters_count", iter_count)
print ('theta: ',theta )
print ('final loss: ', loss)
print ('iters: ', iter_count)

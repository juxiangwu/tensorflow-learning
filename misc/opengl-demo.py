#-*- coding: utf-8 -*-from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import *
import sys

def display():
    glClearColor(0.0,0.0,0.0,0.0)  #R,G,B=(0,0,0)=black , Alpha=0
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(0.0,1.0,0.0);        #R,G,B=(0,1,0)=Green
    glRectf(-0.5,-0.5,0.5,0.5)     #Draw a green Rectangle
    glFlush()                      #将以上的图形绘制到窗口上

glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE|GLUT_RGBA)  #绘图模式，单缓冲，RGBA颜色模式
glutInitWindowSize(500,500)                 # size
glutInitWindowPosition(100,100)             # position 位置
glutCreateWindow(b"simple")                  # 建立视窗(并返回一个ID)
glutDisplayFunc(display)                    # 注册用于绘图的回调函数
glutMainLoop()                              #进入主循环，期间将运行注册的回调函数
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#求fx的函数值
def fx(x, y):
    return x ** 2 - y ** 2
 
def gradient_descent():
    times = 100 # 迭代次数
    alpha = 0.05 # 学习率
    x = 0.5 # x的初始值
    y = 0 # y的初始值
 
    fig = Axes3D(plt.figure()) # 将画布设置为3D
    axis_x = np.linspace(-1, 1, 100)#设置X轴取值范围
    axis_y = np.linspace(-1, 1, 100)#设置Y轴取值范围
    axis_x, axis_y = np.meshgrid(axis_x, axis_y) #将数据转化为网格数据
    z = fx(axis_x,axis_y)#计算Z轴数值
    fig.set_xlabel('X', fontsize=14)
    fig.set_ylabel('Y', fontsize=14)
    fig.set_zlabel('Z', fontsize=14)
    fig.view_init(elev=60,azim=300)#设置3D图的俯视角度，方便查看梯度下降曲线
    fig.plot_surface(axis_x, axis_y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow')) #作出底图
    
    for i in range(times):
        x1 = x        
        y1 = y         
        f1 = fx(x, y)  
        #print("第%d次迭代：x=%f，y=%f，fxy=%f" % (i + 1, x, y, f1))
        x = x - alpha * 2 * x
        y = y + alpha * 2 * y
        f = fx(x, y)
        fig.plot([x1, x], [y1, y], [f1, f], 'ko', ls='-')
    plt.show()
 
if __name__ == "__main__":
    gradient_descent()
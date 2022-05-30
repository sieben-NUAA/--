from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sympy 

lr=0.01 #学习率
xx,yy = sympy.symbols('x y')
zz=xx**2-yy**2 #设置函数
dzdx=sympy.diff(zz, xx) #函数求偏导
dzdy=sympy.diff(zz, yy)
x_val=0.5 #设置初始点
y_val=0
last_gx=0 #保存上一次的梯度
last_gy=0
result=np.array([[0.5,0]]) #记录梯度下降的点
for epoch in range(122): #迭代次数
    grad_x=dzdx.subs({xx: x_val, yy: y_val}) #给偏导函数赋值计算x方向上的梯度
    grad_y=dzdy.subs({xx: x_val, yy: y_val})
    x_val=x_val-(grad_x+last_gx*0.9)*lr #根据梯度更新
    y_val=y_val-(grad_y+last_gy*0.9)*lr
    #注释下面两行就是单纯梯度下降的算法
    #last_gx=grad_x
    #last_gy=grad_y
    result=np.append(result,[[x_val,y_val]],axis=0) #记录新的梯度下降的点
    #print(x_val,y_val)
    #print("-------------------------------------------")

print(grad_x,grad_y) #打印最终的梯度
#print(result)

#画函数3D曲面图
fig = plt.figure(figsize=(8, 6), dpi=100) #开一个窗口进行显示
ax = Axes3D(fig)
u=np.linspace(-1,1,600)
x,y=np.meshgrid(u,u) #x,y生成网格，画3D曲面图需要
print(x.shape)
z=x**2-y**2
ax.plot_surface(x,y,z,rstride=6,cstride=6,cmap=plt.get_cmap('rainbow'),alpha=0.5) #画3D曲面图，alpha调透明度

#画梯度下降3D曲线图
xp=result[:,0] #读取梯度下降点的x坐标
yp=result[:,1]
zp=xp**2-yp**2 #计算梯度下降的点的z坐标
print(zp.shape)
ax.plot(xp, yp, zp, 'o', c='r',linestyle="-", linewidth=2.5) #画3D曲线图
plt.title("3D-picture")# 设置标题

##画函数等高线图
#plt.figure(figsize=(8, 6), dpi=80)
#plt.contourf(x, y, z, 8, alpha=0.75, cmap=plt.cm.hot)
#C = plt.contour(x, y, z, 10, colors = 'black', linewidth = 0.5)
#plt.clabel(C, inline = True, fontsize = 10)# 显示各等高线的数据标签

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Calculate points on Bezier curve
def _bezier(p1, p2, vp1, t):
    p = (1-t)**2*p1 + 2*(1-t)*t*vp1 + t**2*p2
    return p

# Calculate average distance between two points for a specified [p1, p2, p3, n]
def _calculate_average_dist(x1, y1, x2, y2, vx1, vy1, n):
    # calculate the total distance of a normal bezier line 
    # then calculate the average distance between two points
    dist_total = np.sqrt((x1*x1) + (y1*y1))
    t_tmp = 0.0
    x_former = 0.0
    y_former = 0.0
    for j in range(1, n+1):
        t_tmp = t_tmp + 1.0/n
        x_tmp = _bezier(x1,x2,vx1,t_tmp)
        y_tmp = _bezier(y1,y2,vy1,t_tmp)
        dist_total += np.sqrt((x_tmp-x_former)*(x_tmp-x_former) + (y_tmp-y_former)*(y_tmp-y_former))
        x_former = x_tmp
        y_former = y_tmp

    dist_aver = dist_total / n
    return dist_aver
  

# Modify the distance between points to be even
def _bezier_linearlen(x1, y1, x2, y2, vx1, vy1, t, n=50):

    if (n<4):
        return t
    if (t<=0.0) or (t>=1.0):
        return t  


    ll = []
    x = y = 0.0
    px = py = 0
    tt = ni = 0
    i = 0

    ni = 1.0/n
    tt = 0.0

    px = _bezier(x1,x2,vx1,0.0)
    py = _bezier(y1,y2,vy1,0.0)
    ll.insert(0,0.0)

    # calculate all the points
    # ll[] is ???
    for i in range(1, n+1 ):
        tt = tt + ni
        x = _bezier(x1,x2,vx1,tt)
        y = _bezier(y1,y2,vy1,tt)
        ll.insert(i, ll[i-1] + np.sqrt((x-px)*(x-px) + (y-py)*(y-py)))
        px = x
        py = y

    x = 1.0/ll[n]

    for i in range(1, n+1 ):
        ll[i] = ll[i]*x

    for i in range(n):
        if (t>=ll[i])and(t<=ll[i+1]):
            break
        if (i>=n):
            return t
   # print("i=", i)
    x = (ll[i+1]-ll[i])
    if (x<0.0001):
        x = 0.0001       
    x = (t-ll[i]) / x                   
    return (i*(1.0-x) + (i+1)*x) * ni 

# Set the control points. 
# p1(start point)
# p2(end point)
# vp1(intermediate control points)
x1 = 0.0
y1 = 0
p1 = np.array((x1, y1))
x2 = 0
y2 = 10
p2 = np.array((x2, y2))
vx1 = 6
vy1 = 3
vp1 = np.array((vx1, vy1))

# Plot the control points on figure
pxy = np.vstack((p1, vp1, p2)).T

############ Search a proper n ###############
# Receive a specified average distance
aver_dist_fix = 0.5
n_fix = 0
# Set total points on bezier line
for n in range(5, 105, 5):
    aver_dist = _calculate_average_dist(x1, y1, x2, y2, vx1, vy1, n)
    if (abs(aver_dist - aver_dist_fix) <= 0.05):
        n_fix = n
        break
    if (n >= 105):
        print("No proper number of bezier points found!!!")

############ Search a proper n ###############


fig = plt.figure(figsize=(5,5))
plt.plot(pxy[0, :], pxy[1, :], "bo-")

ims=[]
for i in range(n_fix + 1):
    # tt from 0 to 1 with a same interval
    tt = i/n
    # Modify tt to be nonliner in range (0, 1, n)
    tt = _bezier_linearlen(x1,y1,x2,y2,vx1,vy1,tt)
    # until here, tt has been changed to be no linear
    # so that _bezier can generation
    dx = _bezier(x1,x2, vx1, tt)
    dy = _bezier(y1,y2, vy1, tt)
    im = [dx, dy]
    # im is one of the points on bezier_line
    ims.append(im)
    if (i == 1):
       Dist_01 = np.sqrt((im[0]-0)**2 + (im[1]-0)**2)
       #print('Distance(0, 1) =', Dist_01)
    if (i == (1+n//2)):
       Dist_mid = np.sqrt((im[0]-ims[i-1][0])**2 + (im[1]-ims[i-1][1])**2)
       #print('Distance(n/2, 1+n/2) =', Dist_mid)
    if (i == n):
       Dist_final = np.sqrt((im[0]-ims[i-1][0])**2 + (im[1]-ims[i-1][1])**2)
       #print('Distance(n, n-1) =', Dist_final)
    #print('im',i,'=' ,im, '\n')   

#print('Accuracy_1 = ', (Dist_01-Dist_mid)/Dist_mid)
#print('Accuracy_2 = ', (Dist_final-Dist_mid)/Dist_mid)
xsys = np.array(ims).T
plt.plot(xsys[0, :], xsys[1, :], "ro-")
plt.show()

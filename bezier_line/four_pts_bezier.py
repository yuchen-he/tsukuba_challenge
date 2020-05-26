import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _bezier(p1, p2, vp1, vp2, t):
    p = (1-t)**3 * p1 + 3 * (1-t)**2 * t * vp1 + 3 * (1-t) * t**2 * vp2 + t**3 * p2
    return p


def _bezier_linearlen(x1, y1, x2, y2, vx1, vy1, vx2, vy2, t, n=16):
   
    ll = []
    x = y = 0.0
    px = py = 0
    tt = ni = 0
    i = 0


    if (n<4):
        return t


    if (t<=0.0) or (t>=1.0):
        return t

    ni = 1.0/n
    tt = 0.0

    px = _bezier(x1,x2,vx1,vx2,0.0)
    #print('px = ', px)
    py = _bezier(y1,y2,vy1,vy2,0.0)
    ll.insert(0,0.0)

    for i in range(1, n+1 ):
        tt = tt + ni
        x = _bezier(x1,x2,vx1,vx2,tt)
        y = _bezier(y1,y2,vy1,vy2,tt)
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

    x = (ll[i+1]-ll[i])
    if (x<0.0001):
        x = 0.0001       
    x = (t-ll[i]) / x                   
    return (i*(1.0-x) + (i+1)*x) * ni 

x1 = 0
y1 = 0
p0 = np.array((0, 0))
x2 = 0
y2 = 10
p1 = np.array((0, 10))
vx1 = 5
vy1 = 2
p2 = np.array((5, 2))
vx2 = 5.55
vy2 = 4
p3 = np.array((5.55, 4))

pxy = np.vstack((p0, p2, p3, p1)).T
#print('pxy[0, :] =', pxy[0, :])

#n = 100
n = 20

fig = plt.figure(figsize=(5,5))
plt.plot(pxy[0, :], pxy[1, :], "bo-")

ims=[]
for i in range(n+1):
    tt = i/n
    tt = _bezier_linearlen(x1,y1,x2,y2, vx1,vy1,vx2,vy2,tt)
    #print('tt_2 = ', tt)
    dx = _bezier(x1,x2, vx1,vx2, tt)
    dy = _bezier(y1,y2, vy1,vy2, tt)
    #im = plt.scatter(dx, dy)
    im = [dx, dy]
    #print('im = ', im)
    ims.append(im)

xsys = np.array(ims).T
plt.plot(xsys[0, :], xsys[1, :], "ro-")
#ani = animation.ArtistAnimation(fig, ims, interval=0.1, repeat_delay=1000)
plt.show()

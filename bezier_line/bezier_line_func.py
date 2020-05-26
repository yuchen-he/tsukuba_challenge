#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate, misc


def bernstein_basic( t, index_of_cpoint, cpoint_num):
    '''
    複数制御点ベジェ曲線の元であるバーンスタイン基数を返す関数
    -入力
        t: [0,1]のfloat, bezier曲線の位置を示す媒介変数
        index_of_cpoint: 何番目の制御点か？
        cpoint_num:      制御点の総数(始点、終点を含む）
    -出力
        バーンスタイン基数(係数的なもの)
    '''
    return misc.comb( cpoint_num - 1, index_of_cpoint)*\
            t    ** index_of_cpoint *\
            (1-t)** (cpoint_num - 1 - index_of_cpoint)


def bezier_np_pos( t, control_points):
    '''
    ある位置tにあるベジェ曲線のx,y座標を返す関数
    -入力
        t: [0,1]のfloat, bezier曲線の位置を示す媒介変数
        control_points: 制御点を入れたnumpy配列、 shape = (m,2), m=制御点数
    -出力
        x,y座標、shape=(2)の配列
    '''
    pos = np.zeros(2)
    c_point_num = len(control_points)
    for index_of_cpoint in range( c_point_num ): 
        pos+= bernstein_basic( t, index_of_cpoint, c_point_num) * control_points[index_of_cpoint]
    return pos


def bezier_length_calc(control_points, n):
    '''
    制御点からベジェ曲線の長さを返す関数
    -入力
        n: ベジェ曲線上の点数
        control_points: 制御点を入れたnumpy配列、 shape = (m,2), m=制御点数
    -出力
        i+1点目までの曲線長さを入れたlist、shape = (n+1) 
　　　　 例えば、[0, 0.4, 0.74, 1.06, ..., 15.2]、　indexがi 
    '''

    ll = []
    t = i = 0

    if n == 0:
        ni = 0
    else:
        ni = 1.0/n

    p_tmp1 = bezier_np_pos(0.0, control_points)
    ll.insert(0,0.0)

    for i in range(1, n+1):
        t = t + ni
        p_tmp2 = bezier_np_pos(t, control_points)
        ll.insert(i, ll[i-1] + np.sqrt((p_tmp2[0]-p_tmp1[0])**2 + (p_tmp2[1]-p_tmp1[1])**2))
        p_tmp1[0] = p_tmp2[0]
        p_tmp1[1] = p_tmp2[1]

    #print("t=", t, "len(ll)=", len(ll))
    return ll


def _bezier_linearlen(t, control_points, n):

    if (n<4):
        return t
    if (t<=0.0) or (t>=1.0):
        return t  

    ll = bezier_length_calc(control_points, n)
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
    return (i*(1.0-x) + (i+1)*x) / n


def periodic_bezier_np_pos(control_points, dl):
    '''
    概要:
        -任意個数の制御点と点間隔を入力とし、
        -指定された点間隔のxy座標のベジェ曲線点列を返す関数
    入力:
        -control_points => (n,2)の配列、n個の制御点
        -dl             => 点間隔 [m]
    出力:
        - (m,3)のnp配列, (x,y,deg), mは点列数
    '''
    c_point_num = len( control_points )
    target_points = np.empty((0,3))
    target_rad = []

    # ベジェ曲線の長さを計算するため、
    # 制御点の始点と終点の直線距離から、必要な点数を大まかなに設定する
    ll_direct = np.sqrt((control_points[c_point_num-1][1] - control_points[0][1]) **2 \
                        + (control_points[c_point_num-1][0] - control_points[0][0]) **2 )
    n_tmp = int(ll_direct // dl)
    ll = bezier_length_calc(control_points, n=n_tmp)
    n_fix = int(ll[n_tmp] // dl)

    cur_point = control_points[0]

    for i in range(1, n_fix +1):

        pre_point = cur_point

        # Modify t to be nonliner in range (0, 1, n_fix)
        t = float(i)/n_fix
        t = _bezier_linearlen(t, control_points, n=n_fix)

        cur_point = bezier_np_pos(t, control_points)
        pre_rad = math.degrees( math.atan((cur_point[1]-pre_point[1]) / (cur_point[0]-pre_point[0])) )
        pre_points = np.hstack( [pre_point, pre_rad] ) 
        target_points = np.vstack( [target_points, pre_points] )     

    # n_fix個目のradをn_fix+1個目にそのまま与える
    if(len(target_points) != 0):
        pre_points = np.hstack( [cur_point, pre_rad] )
        target_points = np.vstack( [target_points, pre_points] )    

    return target_points


if __name__ == '__main__':

    #Debug
    cp0 = np.array([ 0.0,   0.0] )
    cp1 = np.array([ 3.0,  10.0] )
    #cp1 = np.array([ 0.0,  0.0] )
    cp2 = np.array([ 15.0, -10.0] )
    cp3 = np.array([ 20.0,  15.0] )
    cps = np.vstack((cp0,cp1,cp2,cp3)).T
    #cps = np.vstack((cp0,cp1,cp3)).T

    plt.figure()
    plt.plot(cps[0,:],cps[1,:],"bo-")

    control_points = [cp0,cp1,cp2,cp3]
    #control_points = [cp0,cp1,cp3]
    xsys = periodic_bezier_np_pos(control_points = control_points, dl= 0.5)    
    print("Number of points on bezier_line is ", len(xsys))    
    print(xsys)

    plt.plot(xsys.T[0,:], xsys.T[1,:], "ro-")
    plt.axis('equal')
    plt.show()


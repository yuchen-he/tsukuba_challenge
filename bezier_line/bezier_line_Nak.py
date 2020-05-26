#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from scipy import interpolate, misc
import numpy as np
from matplotlib import pyplot as plt

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

def diff_bernstein_basic( t, index_of_cpoint, cpoint_num):
    '''
    複数制御点ベジェ曲線の元であるバーンスタイン基数のt微分を返す関数
    -入力
        t: [0,1]のfloat, bezier曲線の位置を示す媒介変数
        index_of_cpoint: 何番目の制御点か？
        cpoint_num:      制御点の総数(始点、終点を含む）
    -出力
        バーンスタイン基数(係数的なもの)のt微分
    -注意点
        始点と終点のinudex_of_cpointの場合、微分は0divが発生するため、if文で分けた
    '''
    keisu   = misc.comb( cpoint_num - 1, index_of_cpoint)
    if index_of_cpoint != 0 and cpoint_num != (index_of_cpoint+1):
        diff1 = ( index_of_cpoint * t ** ( index_of_cpoint - 1 ) ) \
                  * (1.0-t)**(cpoint_num - 1 - index_of_cpoint)
        diff2 = ( t    ** index_of_cpoint ) \
                  * -1.0 * (cpoint_num - 1 - index_of_cpoint) * (1-t)**(cpoint_num - 2 - index_of_cpoint)
        return keisu * ( diff1+diff2 )
    elif index_of_cpoint == 0:
        diff2 = \
                  -1.0 * (cpoint_num - 1 - index_of_cpoint) * (1.0-t)**(cpoint_num - 2 - index_of_cpoint)
        return keisu * diff2
    elif cpoint_num == (index_of_cpoint+1):
        diff1 = ( index_of_cpoint * t ** ( index_of_cpoint - 1 ) ) 
        return keisu * diff1
    else:
        print("something wrong")
                

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

def diff_bezier_np_pos( t, control_points):
    '''
    ある位置tにあるベジェ曲線のx,y座標のt微分を返す関数
    -入力
        t: [0,1]のfloat, bezier曲線の位置を示す媒介変数
        control_points: 制御点を入れたnumpy配列、 shape = (m,2), m=制御点数
    -出力
        x,y座標のt微分値、shape=(2)の配列
    '''
    pos = np.zeros(2)
    c_point_num = len(control_points)
    for index_of_cpoint in range( c_point_num ): 
        pos+= diff_bernstein_basic( t, index_of_cpoint, c_point_num) * control_points[index_of_cpoint]
    return pos

def periodic_bezier_np_pos( control_points, dl):
    '''
    概要:
        -任意個数の制御点と点間隔を入力とし、
        -指定された点間隔のxy座標のベジェ曲線点列を返す関数
    入力:
        -control_points => (n,2)の配列、n個の制御点
        -dl             => 点間隔 [m]
    出力:
        - (m,3)のnp配列, (x,y,deg), mは点列数
    TODO:
        - 最後の1点の追加(t=1.0)は点間隔によらず置くので
        - その部分だけ間隔が短くなっている
    '''
    # t = [0.0, 1.0]
    # initial_t = 0.0
    # next_t    = prev_t +  dl / diff_dist
    c_point_num = len( control_points )
    
    cur_t  = 0.0
    prev_t = 0.0
    target_points = np.empty((0,3))
    
    # tが1.0に到達するまで繰り返しポイントを求める
    while cur_t < 1.0:
        # calculate dx/dt, dy/dt for extractiong angle
        # calculate diff_dist for extracting next t
        prev_t  = cur_t
        
        # 現在のtでのdx/dt, dy/dtの算出
        diff_xy = diff_bezier_np_pos( prev_t, control_points )
        # angle
        tmp_rad   = math.degrees( math.atan2( diff_xy[1], diff_xy[0] ) )
        # diff_dist
        diff_dist = math.sqrt( diff_xy[0] ** 2 + diff_xy[1] ** 2 )
        
        # append new point at t
        cur_point = bezier_np_pos( cur_t, control_points )
        cur_point = np.hstack( [cur_point, tmp_rad] )
        target_points = np.vstack( [ target_points, cur_point ] )
        
        # update cur_t
        cur_t = prev_t + dl / diff_dist
        
    # append last point at t=1.0
    # 現在のtでのdx/dt, dy/dtの算出
    diff_xy = diff_bezier_np_pos( prev_t, control_points )
    # angle
    tmp_rad   = math.degrees( math.atan2( diff_xy[1], diff_xy[0] ) )
    # x,y
    cur_point = bezier_np_pos( 1.0, control_points)
    # append
    cur_point = np.hstack( [cur_point, tmp_rad] )
    target_points = np.vstack( [ target_points, cur_point ] )
    return target_points


if __name__ == '__main__':
    # Debug
    cp0 = np.array([  0.0,   0.0] )
    cp1 = np.array([ 20.0,  20.0] )
    cp2 = np.array([ 40.0, -20.0] )
    cp3 = np.array([100.0,-100.0] )
    cps = np.vstack( [cp0,cp1,cp2,cp3] )
    # unit function test
    '''
    points        = np.empty( (0,2) )
    d_points      = np.empty( (0,2) )
    rads          = np.empty( (0,1) )
    for t in np.linspace(0.0, 1.0, 100):
        #print t
        tmp_p        = bezier_np_pos( t, cps)
        tmp_dp       = diff_bezier_np_pos( t, cps)
        tmp_rad      = math.degrees( math.atan2( tmp_dp[1], tmp_dp[0] ) )
        points   = np.vstack( [ points,    tmp_p  ] )
        d_points = np.vstack( [ d_points,  tmp_dp ] )
        rads     = np.vstack( [ rads,      tmp_rad] )
    '''
    
    # target function test
    points = periodic_bezier_np_pos( cps, 0.25)
    for index in range(len(points)-1):
        dp = points[index+1] - points[index]
        dist = math.sqrt( dp[0]**2 + dp[1]**2 )
        #print dist
        
    #print points
    plt.figure()
    plt.plot( points[:,0], points[:,1], "ro-")
    plt.axis('equal')
    plt.show()

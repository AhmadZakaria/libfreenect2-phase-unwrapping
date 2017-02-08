# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:55:28 2016

@author: ahmad
"""

import multiprocessing as mp

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d.axes3d as axes3d


# TODO: bin the phase values, and memoize them with their corresponding distance (or index)
def common_element(array1, array2, array3):
    set1 = set(array1)
    set2 = set(array2)
    set3 = set(array3)
    common = set1.intersection(set2).intersection(set3)
    max_iter = 10
    while (len(common) < 1 and max_iter > 0):
        tmp = set([])
        for i in set1:
            tmp.add(i + 1)
            tmp.add(i - 1)
        set1 = set1.union(tmp)
        tmp = set([])
        for i in set2:
            tmp.add(i + 1)
            tmp.add(i - 1)
        set2 = set2.union(tmp)
        tmp = set([])
        for i in set3:
            tmp.add(i + 1)
            tmp.add(i - 1)
        set3 = set3.union(tmp)
        common = set1.intersection(set2).intersection(set3)
        max_iter = max_iter - 1
    if len(common) > 0:
        return common.pop()
    return -1

def common_element2(array1, array2, array3):
    diffs = np.zeros((len(array1),len(array2),len(array3)), dtype=np.float)
    means_ = np.zeros((len(array1),len(array2),len(array3)), dtype=np.float)
#    means__ = np.asarray(array1, dtype=np.float64)
#    means__ = (array1 + array2 + array3) / 3.0
#    diffs__ = np.abs(array1-means__)+ np.abs(array2-means__)+ np.abs(array3-means__)  
#    
    for i0, p0 in enumerate(array1):
        for i1, p1 in enumerate(array2):
            for i2, p2 in enumerate(array3):
                mean_ = (p0+p1+p2)/3.0               
                diffs[i0,i1,i2] = np.abs(p0-p1)+np.abs(p1-p2)+np.abs(p2-p0)
                means_[i0,i1,i2] = mean_
#    print "np.argmin(diffs): ", np.argmin(diffs.flatten())
    return int(means_[np.where(diffs == diffs.min())][0]), diffs[np.where(diffs == diffs.min())][0]

unwrapped = []
freq1 = 10
freq2 = 2
freq3 = 15
period = 18.5
# samples_n = freq3 * period * 100
samples_n = 10000
dist = np.linspace(0, 18.5, samples_n)
phase1 = (signal.sawtooth(2 * np.pi * freq1 / period * dist) + 1) * np.pi
phase2 = (signal.sawtooth(2 * np.pi * freq2 / period * dist) + 1) * np.pi
phase3 = (signal.sawtooth(2 * np.pi * freq3 / period * dist) + 1) * np.pi


def do_stuff(p0, p1, p2, row, col):
    near1 = nearest_k(phase1, p0, 10)
    near2 = nearest_k(phase2, p1, 2)
    near3 = nearest_k(phase3, p2, 15)

    res = common_element(near1, near2, near3)
    return [row, col, res]


def nearest_15(array, value):
    indices = np.abs(array - value).argsort()
    if (indices.size > 15):
        indices = indices[0:15]
    return indices


def nearest_k(array, value, k):
    indices = np.abs(array - value).argsort()
    wave_length = (samples_n/k)
    first_idx = indices[0]% wave_length
    return [first_idx + wave_length* i for i in range(k)]


def unwrap_superior1(phases, p0tables):
    global unwrapped
    # phase1 = (signal.sawtooth(2 * np.pi * freq1 / period * dist) + 1) * np.pi
    # phase2 = (signal.sawtooth(2 * np.pi * freq2 / period * dist) + 1) * np.pi
    # phase3 = (signal.sawtooth(2 * np.pi * freq3 / period * dist) + 1) * np.pi

    #    fig = plt.figure()

    unwrapped = np.zeros((phases.shape[0], phases.shape[1]), dtype=np.float)
    pool = mp.Pool(processes=8)
    counter = 0
    for row in range(phases.shape[0]):
        for col in range(phases.shape[1]):
            p0 = phases[row, col, 0]
            p1 = phases[row, col, 1]
            p2 = phases[row, col, 2]

            # near1 = nearest_15(phase1, p0)
            # near2 = nearest_15(phase2, p1)
            # near3 = nearest_15(phase3, p2)

            # async_result = pool.apply_async(common_element, args=(near1, near2, near3, row, col),
            #                                 callback=proc_res)  # tuple of args for foo

            async_result = pool.apply_async(do_stuff, args=(p0, p1, p2, row, col),
                                            callback=proc_res)  # tuple of args for foo

            counter += 1
            # print counter
            # if (counter == 15):
            #     exit()
            #            tmp = common_element(near1, near2, near3)
            # unwrapped[row, col] = async_result.get()
    pool.close()
    pool.join()
    return unwrapped


def unwrap_superior(phases):
    samples = 100
    period = 2.0 * np.pi
    step = period / samples
    unwrapped = np.zeros((phases.shape[0], phases.shape[1]), dtype=np.float)
    confidence = np.zeros((phases.shape[0], phases.shape[1]), dtype=np.float)

    #    nearest10_80mhz = np.zeros(shape=(samples, 10), dtype=np.float)
    #    nearest2_16mhz = np.zeros(shape=(samples, 2), dtype=np.float)
    #    nearest15_120mhz = np.zeros(shape=(samples, 15), dtype=np.float)
    #    with open("lookup.dat", 'r') as lookup:
    #        npzfile = np.load(lookup)
    #        nearest10_80mhz = npzfile['nearest10_80mhz']
    #        nearest2_16mhz = npzfile['nearest2_16mhz']
    #        nearest15_120mhz = npzfile['nearest15_120mhz']

    common_list = np.load("common1.dat")
    confidence_map = np.load("confidence_map.dat")
    
    print "common_list",common_list.shape, common_list.max(), common_list.min()

    for row in range(phases.shape[0]):
        for col in range(phases.shape[1]):
            p0 = phases[row, col, 0]
            p1 = phases[row, col, 1]
            p2 = phases[row, col, 2]

#            i0 = int(round(p0 / step))-1
#            i1 = int(round(p1 / step))-1
#            i2 = int(round(p2 / step))-1
            i0 = int((p0 / step))
            i1 = int((p1 / step))
            i2 = int((p2 / step))          
#            print ((i0+i1+i2)/3.0)
#            print p0, p1, p2
#            print i0, i1, i2
            unwrapped[row, col] = common_list[i0, i1, i2]
            confidence[row, col] = confidence_map[i0, i1, i2]
    return unwrapped,confidence


def unwrap_superior_newidx(phases):
    samples = 100
    period = 2.0 * np.pi
    step = period / samples
    unwrapped = np.zeros((phases.shape[1], phases.shape[2]), dtype=np.float)
    confidence = np.zeros((phases.shape[1], phases.shape[2]), dtype=np.float)

    #    nearest10_80mhz = np.zeros(shape=(samples, 10), dtype=np.float)
    #    nearest2_16mhz = np.zeros(shape=(samples, 2), dtype=np.float)
    #    nearest15_120mhz = np.zeros(shape=(samples, 15), dtype=np.float)
    #    with open("lookup.dat", 'r') as lookup:
    #        npzfile = np.load(lookup)
    #        nearest10_80mhz = npzfile['nearest10_80mhz']
    #        nearest2_16mhz = npzfile['nearest2_16mhz']
    #        nearest15_120mhz = npzfile['nearest15_120mhz']

    common_list = np.load("common1.dat")
    confidence_map = np.load("confidence_map.dat")
    
    print "common_list",common_list.shape, common_list.max(), common_list.min()

    for row in range(phases.shape[1]):
        for col in range(phases.shape[2]):
            p0 = phases[0, row, col]
            p1 = phases[1, row, col]
            p2 = phases[2, row, col]

#            i0 = int(round(p0 / step))-1
#            i1 = int(round(p1 / step))-1
#            i2 = int(round(p2 / step))-1
            i0 = min(int((p0 / step)),common_list.shape[0]-1)
            i1 = min(int((p1 / step)),common_list.shape[0]-1)
            i2 = min(int((p2 / step))   ,common_list.shape[0]-1)       
#            print ((i0+i1+i2)/3.0)
#            print p0, p1, p2
#            print i0, i1, i2
            unwrapped[row, col] = common_list[i0, i1, i2]
            confidence[row, col] = confidence_map[i0, i1, i2]
    return unwrapped,confidence


def proc_res(res):
    global unwrapped
    print res[0], res[1], res[2]
    unwrapped[res[0]][res[1]] = res[2]


#
#    ax1 = fig.add_subplot(3,1,1)
#    ax2 = fig.add_subplot(3,1,2)
#    ax3 = fig.add_subplot(3,1,3)
#    
#    ax1.set_xlim([0, 20])
#    ax1.set_ylim([0, 10])
#    ax2.set_xlim([0, 20])
#    ax2.set_ylim([0, 10])
#    ax3.set_xlim([0, 20])
#    ax3.set_ylim([0, 10])
#    
#    for x in near1:
#        ax1.axvline(x=x/(freq3 *100.0),ymin=-1.2,ymax=1,c="green",linewidth=1, linestyle="dotted",zorder=0, clip_on=False)
#    for x in near2:
#        ax2.axvline(x=x/(freq3 * 100.0),ymin=0,ymax=1.2,c="green",linewidth=1, linestyle="dotted", zorder=0,clip_on=False)
#    for x in near3:
#        ax3.axvline(x=x/(freq3 * 100.0),ymin=0,ymax=1.2,c="green",linewidth=1, linestyle="dotted", zorder=0,clip_on=False)
#    
#    for x in common:
#        ax3.axvline(x=x/(freq3 * 100.0),ymin=0,ymax=1.2,c="red",linewidth=2, linestyle="dashed", zorder=0,clip_on=False)
#    
#    ax1.plot(dist, phase1,c="blue",zorder=1,)
#    ax2.plot(dist, phase2,c="blue",zorder=1)
#    ax3.plot(dist, phase3,c="blue",zorder=1)
#    plt.show()
#    
#    print phase1.max()
#    print near1
#    print phase1[near1]



def cube_marginals(cube, normalize=False):
    c_fcn = np.mean if normalize else np.sum
    xy = c_fcn(cube, axis=0)
    xz = c_fcn(cube, axis=1)
    yz = c_fcn(cube, axis=2)
    return(xy,xz,yz)

def plotcube(cube,x=None,y=None,z=None,normalize=False,plot_front=False):
    """Use contourf to plot cube marginals"""
    (Z,Y,X) = cube.shape
    (xy,xz,yz) = cube_marginals(cube,normalize=normalize)
    if x == None: x = np.arange(X)
    if y == None: y = np.arange(Y)
    if z == None: z = np.arange(Z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # draw edge marginal surfaces
    offsets = (Z-1,0,X-1) if plot_front else (0, Y-1, 0)
    cset = ax.contourf(x[None,:].repeat(Y,axis=0), y[:,None].repeat(X,axis=1), xy, zdir='z', offset=offsets[0], cmap=plt.cm.coolwarm, alpha=0.75)
    cset = ax.contourf(x[None,:].repeat(Z,axis=0), xz, z[:,None].repeat(X,axis=1), zdir='y', offset=offsets[1], cmap=plt.cm.coolwarm, alpha=0.75)
    cset = ax.contourf(yz, y[None,:].repeat(Z,axis=0), z[:,None].repeat(Y,axis=1), zdir='x', offset=offsets[2], cmap=plt.cm.coolwarm, alpha=0.75)

    # draw wire cube to aid visualization
    ax.plot([0,X-1,X-1,0,0],[0,0,Y-1,Y-1,0],[0,0,0,0,0],'k-')
    ax.plot([0,X-1,X-1,0,0],[0,0,Y-1,Y-1,0],[Z-1,Z-1,Z-1,Z-1,Z-1],'k-')
    ax.plot([0,0],[0,0],[0,Z-1],'k-')
    ax.plot([X-1,X-1],[0,0],[0,Z-1],'k-')
    ax.plot([X-1,X-1],[Y-1,Y-1],[0,Z-1],'k-')
    ax.plot([0,0],[Y-1,Y-1],[0,Z-1],'k-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim([0,cube.max()/30.])
    ax.set_ylim([0,cube.max()/30.])
    ax.set_zlim([0,cube.max()/30.])
    plt.show()

if __name__ == "__main__":
    samples = 100
    period = 2 * np.pi
    step = period / samples

    nearest10_80mhz = np.zeros(shape=(samples, 10), dtype=np.float)
    nearest2_16mhz = np.zeros(shape=(samples, 2), dtype=np.float)
    nearest15_120mhz = np.zeros(shape=(samples, 15), dtype=np.float)

#    for i in range(samples):
#        nearest10_80mhz[i] = nearest_k(phase1, i * step, 10)
#        nearest2_16mhz[i] = nearest_k(phase2, i * step, 2)
#        nearest15_120mhz[i] = nearest_k(phase3, i * step, 15)
#    with open("lookup.dat", 'w') as lookup:
#        np.savez(lookup, nearest10_80mhz=nearest10_80mhz, nearest2_16mhz=nearest2_16mhz,
#                 nearest15_120mhz=nearest15_120mhz)
#    with open("lookup.dat", 'r') as lookup:
#        npzfile = np.load(lookup)
#        nearest10_80mhz = npzfile['nearest10_80mhz']
#        nearest2_16mhz = npzfile['nearest2_16mhz']
#        nearest15_120mhz = npzfile['nearest15_120mhz']
#    
#    common_list = np.zeros(shape=(samples, samples, samples), dtype=np.int32)
#    confidence_map = np.zeros(shape=(samples, samples, samples), dtype=np.int32)
#    for i0, p0 in enumerate(nearest10_80mhz):
#        for i1, p1 in enumerate(nearest2_16mhz):
#            for i2, p2 in enumerate(nearest15_120mhz):
#                common_list[i0, i1, i2], confidence_map[i0, i1, i2] = common_element2(p0, p1, p2)
#    with open("common1.dat", 'w') as common:
#        np.save(common, common_list)
#    with open("confidence_map.dat", 'w') as common:
#        np.save(common, confidence_map)
    

    common_list = np.load("common1.dat")
    confidence_map = np.load("confidence_map.dat")
    plt.figure("confidence_map[:, :, 0]")
    plt.imshow(confidence_map[:, :, 0])
    plt.figure("confidence_map[:, 0, :]")
    plt.imshow(confidence_map[:, 0, :])
    plt.figure("confidence_map[0, :, :]")
    plt.imshow(confidence_map[0, :, :])

    plt.figure("common_list[:, :, 0]")
    plt.imshow(common_list[:, :, 0])
    plt.figure("common_list[:, 0, :]")
    plt.imshow(common_list[:, 0, :])
    plt.figure("common_list[0, :, :]")
    plt.imshow(common_list[0, :, :])
    
#    plotcube(common_list)
#    plotcube(confidence_map)
#    common_list = np.load("common.dat")
#
#    p0 = 2 * np.pi
#    p1 = 2 * np.pi
#    p2 = 2 * np.pi
#
#    i0 = int(p0 * step)
#    i1 = int(p1 * step)
#    i2 = int(p2 * step)

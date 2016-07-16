# -*- coding: utf-8 -*-
'''
Created on Mon Jun 13 12:32:36 2016

@author: ahmad
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as clrs
from scipy.misc import imread
import scipy.constants as csts
import numpy as np
from math import atan2, sin, cos, floor, fabs, log, exp, sqrt
#from testing_sawtooth import *

from skimage.restoration import unwrap_phase
from skimage import exposure
from skimage import filter
from scipy.signal import medfilt
from scipy.optimize import minimize

ccp = 'Greys_r'
cca = 'Greens_r'

def interpolate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def intersect_saw  (freq, num_wraps_freq_1, val, num_points, sigma):
    #val ist the middle point

    intersects = np.ndarray((num_wraps_freq_1,num_points), dtype=np.float)
    points=np.ndarray((num_points), dtype=np.float)
    middle_point=int (floor (num_points/2))
    first_dist= csts.c * (2*np.pi) /(4*np.pi*freq*1000000)        #distance at which the first unwrap will happend
    
    #print "first dist of", freq, " is ", first_dist
    
    #print "middle point is", middle_point
    #print "val is ", val
    
    #fill the values for points
    points[middle_point]=val
    #for higher values
    for i in range (middle_point+1,int(num_points)):
        #print "filling high value at index", i
        dist_middle=abs(i-middle_point)
        #print "dist to middle is ", dist_middle
        points[i]=val + sigma*dist_middle
        if points[i]>2*np.pi:
            points[i]=points[i]-2*np.pi
        #print "high val is", points[i]
    for i in range (0,middle_point):
        dist_middle=abs(i-middle_point)
        points[i]=val - sigma*dist_middle
        if points[i]<0:
            points[i]=2*np.pi + points[i]
        #print "low val is", points[i]
        #print "filling low value at index", i
        
    #now we have in points the values of the phases that we will intersect
    
    #now we get the first intersect of all of them to the first saw tooth, which is just a linear interpolation
    for i  in range (num_points):
        intersects[0,i]=interpolate(points[i],0,2*np.pi,0,first_dist)
        #print "interpolation of ", points[i], " is ", intersects[0,i]
        
    #now the next rows of the intersects matrix is just multiples of the first_dist with respect to the first row
    for i  in range (1,num_wraps_freq_1):
        for col in range (num_points):
            intersects[i,col]=intersects[0,col] + first_dist*i
        
    return intersects
def constraint_funk(x):
    if x[0] <11 and x[1] <3 and x[2] < 16 and (x>0).all():
        return 0
    else :
        return 1
        
def ob_funk(Ns, phases):
    n0, n1, n2 = Ns[0], Ns[1], Ns[2]
    t0 = phases[0] * 3.0 / (2. * np.pi)
    t1 = phases[1] * 15.0 / (2. * np.pi)
    t2 = phases[2] * 2.0 / (2. * np.pi)

    sigma_t_0_sq = ((3. * 2. * np.pi) / (80.)) ** 2
    sigma_t_1_sq = ((15. * 2. * np.pi) / (16.)) ** 2
    sigma_t_2_sq = ((2. * 2. * np.pi) / (120.)) ** 2

    sigma_epsilon_1_sq = sigma_t_1_sq + sigma_t_0_sq

    sigma_epsilon_2_sq = sigma_t_2_sq + sigma_t_0_sq

    sigma_epsilon_3_sq = sigma_t_2_sq + sigma_t_1_sq    
    
    epsilon1_sq = (3 * n0 - 15 * n1 - (t1 - t0))**2
    epsilon2_sq = (3 * n0 - 2 * n2 - (t2 - t0))**2
    epsilon3_sq = (15 * n1 - 2 * n2 - (t2 - t1))**2
    
    result = (epsilon1_sq / sigma_epsilon_1_sq) + (epsilon2_sq / sigma_epsilon_2_sq) + (epsilon3_sq / sigma_epsilon_3_sq)
    
    return result


def min_ob_funk(ob_funk, pixel_phases):
    min_values = [1000, 1000, 1000]   
    min_ret = 100000
    for n0 in range(1,11):
        for n1 in range(1,3):
            for n2 in range(1,16):
                ret = ob_funk([n0, n1, n2], pixel_phases)
                if ret < min_ret:
                    min_ret = ret
                    min_values = [n0, n1, n2]
    return min_values

if __name__ == "__main__":
    print "started"

    #    f = open("/home/ahmad/Documents/Masters/Computational-Photography/project/libfreenect2/build-qt/bin/Frames/tab11to16.dat", "r")
    #    a = np.fromfile(f, dtype=np.uint16)
    #    print a

    with open("p0tables.dat", "rb") as p0tf:
        p0tables = np.fromfile(p0tf, dtype=np.uint16)
        p0tables = np.flipud(p0tables.reshape((424, 512, 3)))
        print p0tables.shape
    #        plt.figure("p0 Tables")
    #        plt.subplot(1, 3, 1)
    #        plt.imshow(p0tables[:,:,0])
    #        plt.subplot(1, 3, 2)
    #        plt.imshow(p0tables[:,:,1])
    #        plt.subplot(1, 3, 3)
    #        plt.imshow(p0tables[:,:,2])

    # with open("decodedIR", "rb") as f2:
#    with open("../data/20160616-120946344.ir", "rb") as f2: # Boris
    with open("../data/20160616-12434139.ir", "rb") as f2:
        plt.figure("Raw data")
        irpic = np.fromfile(f2, dtype=np.int32)
        print (irpic.shape)
        irpic = (abs(((irpic.reshape((9, 424, 512))))))
        print (irpic.shape)

        ax = plt.subplot(4, 1, 1)

        i = imread("dumprgb12.jpeg")
#        i = imread("../data/20160616-120946332.rgb") #Boris
        #i = imread("../data/20160616-12434225.rgb") #Corbin

        ax.set_axis_off()

        plt.imshow(i)
        counter = 4
        for row in range(2, 5):
            for col in range(2, 5):
                # flip once
                irpic[counter - 4][:][:] = np.flipud(irpic[counter - 4][:][:])
                ax = plt.subplot(4, 3, counter)
                ax.set_axis_off()
                plt.imshow(np.minimum(irpic[counter - 4][:][:], np.mean(irpic[counter - 4][:][:])), cmap='inferno')
                counter = counter + 1

    # %%
    ### ACCORDING TO PATENT, Phases and amplitudes
    phases = np.ndarray((424, 512, 3), dtype=np.float)
    amps = np.ndarray((424, 512, 3), dtype=np.float)

    #    first_term = irpic[0, :, :] * np.sin(0) + irpic[1, :, :] * np.sin(2 * np.pi / 3) + irpic[2, :, :] * np.sin(
    #        4 * np.pi / 3)
    #    second_term = irpic[0, :, :] * np.cos(0) + irpic[1, :, :] * np.cos(2 * np.pi / 3) + irpic[2, :, :] * np.cos(
    #        4 * np.pi / 3)
    #
    #    phases[:, :, 0] = np.arctan2(-first_term, second_term)
    #    amps[:, :, 0] = np.sqrt(np.square(second_term) + np.square(first_term)) / 2.
    #
    #    first_term = irpic[3, :, :] * sin(0) + irpic[4, :, :] * sin(2 * np.pi / 3) + irpic[5, :, :] * sin(4 * np.pi / 3)
    #    second_term = irpic[3, :, :] * cos(0) + irpic[4, :, :] * cos(2 * np.pi / 3) + irpic[5, :, :] * cos(4 * np.pi / 3)
    #
    #    phases[:, :, 1] = np.arctan2(-(first_term), (second_term))
    #    amps[:, :, 1] = np.sqrt(np.square(second_term) + np.square(first_term)) / 2.
    #
    #    first_term = irpic[6, :, :] * sin(0) + irpic[7, :, :] * sin(2 * np.pi / 3) + irpic[8, :, :] * sin(4 * np.pi / 3)
    #    second_term = irpic[6, :, :] * cos(0) + irpic[7, :, :] * cos(2 * np.pi / 3) + irpic[8, :, :] * cos(4 * np.pi / 3)
    #
    #    phases[:, :, 2] = np.arctan2(-(first_term), (second_term))
    #    amps[:, :, 2] = np.sqrt(np.square(second_term) + np.square(first_term)) / 2.
    #    # calculate the corrected amplitude according to https://hal.inria.fr/hal-00725654/PDF/TOF.pdf
    #    distance = np.ndarray((424, 512, 3), dtype=np.float)
    #
    #    distance[:, :, 0] = csts.c * phases[:, :, 0] / (4 * np.pi * 80)
    #    distance[:, :, 1] = csts.c * phases[:, :, 1] / (4 * np.pi * 16)
    #    distance[:, :, 2] = csts.c * phases[:, :, 2] / (4 * np.pi * 120)

    # %%

    proc_p0tables = p0tables * -0.000031 * np.pi
    proc_p0tables[:, :, 1] = proc_p0tables[:, :, 1] + 2. * np.pi / 3.
    proc_p0tables[:, :, 2] = proc_p0tables[:, :, 2] + 4. * np.pi / 3.

    cos_p0tables = np.cos(proc_p0tables)
    sin_p0tables = np.sin(-1. * proc_p0tables)

    #    m[0] == 32767 || m[1] == 32767 || m[2] == 32767

    irpic_f = np.asarray(irpic, dtype=np.float32)
    irpic_f = irpic.astype(dtype=np.float32)

    ir_a1 = cos_p0tables[:, :, 0] * irpic_f[0, :, :] + cos_p0tables[:, :, 1] * irpic_f[1, :, :] + cos_p0tables[:, :,
                                                                                                  2] * irpic_f[2, :, :]
    ir_b1 = sin_p0tables[:, :, 0] * irpic_f[0, :, :] + sin_p0tables[:, :, 1] * irpic_f[1, :, :] + sin_p0tables[:, :,
                                                                                                  2] * irpic_f[2, :, :]

    ir_a1 = ir_a1 * 1.322581
    ir_b1 = ir_b1 * 1.322581

    ir_amplitude1 = np.sqrt(ir_a1 ** 2 + ir_b1 ** 2)
    #    phases[:, :, 0] = np.arctan2(ir_b1, ir_a1)


    ir_a1[irpic[0, :, :] == 32767] = 0.
    ir_a1[irpic[1, :, :] == 32767] = 0.
    ir_a1[irpic[2, :, :] == 32767] = 0.

    ir_b1[irpic[0] == 32767] = 0.
    ir_b1[irpic[1] == 32767] = 0.
    ir_b1[irpic[2] == 32767] = 0.

    ir_amplitude1[irpic[0] == 32767] = 0.
    ir_amplitude1[irpic[1] == 32767] = 0.
    ir_amplitude1[irpic[2] == 32767] = 0.

    ir_a2 = cos_p0tables[:, :, 0] * irpic_f[3, :, :] + cos_p0tables[:, :, 1] * irpic_f[4, :, :] + cos_p0tables[:, :,
                                                                                                  2] * irpic_f[5, :, :]
    ir_b2 = sin_p0tables[:, :, 0] * irpic_f[3, :, :] + sin_p0tables[:, :, 1] * irpic_f[4, :, :] + sin_p0tables[:, :,
                                                                                                  2] * irpic_f[5, :, :]

    ir_a2 = ir_a2 * 1.
    ir_b2 = ir_b2 * 1.

    ir_amplitude2 = np.sqrt(ir_a2 ** 2 + ir_b2 ** 2)
    #    phases[:, :, 1] = np.arctan2(ir_b2, ir_a2)



    ir_a2[irpic[0, :, :] == 32767] = 0.
    ir_a2[irpic[1, :, :] == 32767] = 0.
    ir_a2[irpic[2, :, :] == 32767] = 0.

    ir_b2[irpic[0] == 32767] = 0.
    ir_b2[irpic[1] == 32767] = 0.
    ir_b2[irpic[2] == 32767] = 0.

    ir_amplitude2[irpic[3] == 32767] = 0.
    ir_amplitude2[irpic[4] == 32767] = 0.
    ir_amplitude2[irpic[5] == 32767] = 0.

    ir_a3 = cos_p0tables[:, :, 0] * irpic_f[6, :, :] + cos_p0tables[:, :, 1] * irpic_f[7, :, :] + cos_p0tables[:, :,
                                                                                                  2] * irpic_f[8, :, :]
    ir_b3 = sin_p0tables[:, :, 0] * irpic_f[6, :, :] + sin_p0tables[:, :, 1] * irpic_f[7, :, :] + sin_p0tables[:, :,
                                                                                                  2] * irpic_f[8, :, :]

    ir_a3 = ir_a3 * 1.612903
    ir_b3 = ir_b3 * 1.612903

    ir_amplitude3 = np.sqrt(ir_a3 ** 2 + ir_b3 ** 2)
    #    phases[:, :, 2] = np.arctan2(np.real(ir_b3), np.imag(ir_a3))


    ir_a3[irpic[0, :, :] == 32767] = 0.
    ir_a3[irpic[1, :, :] == 32767] = 0.
    ir_a3[irpic[2, :, :] == 32767] = 0.

    ir_b3[irpic[0] == 32767] = 0.
    ir_b3[irpic[1] == 32767] = 0.
    ir_b3[irpic[2] == 32767] = 0.

    ir_amplitude3[irpic[6] == 32767] = 0.
    ir_amplitude3[irpic[7] == 32767] = 0.
    ir_amplitude3[irpic[8] == 32767] = 0.

    ir_amplitude = (ir_amplitude1 + ir_amplitude2 + ir_amplitude3) / 3.

    ir_p = - np.arctan((ir_amplitude1 + ir_amplitude2 + ir_amplitude3))

    #    plt.figure("ir_a")
    #    plt.imshow(ir_a)
    #    plt.figure("ir_b")
    #    plt.imshow(ir_b)
    amps_eq = np.minimum(ir_amplitude, 1000)

    #    plt.figure("ir_amp")
    #    plt.imshow(amps_eq,cmap="Greens_r")
    #    plt.figure("ir_pha")
    #    plt.imshow(ir_p)




    # %%
    pp0 = ((irpic[0, :, :] * np.exp(-1j * (0))))
    pp1 = ((irpic[1, :, :] * np.exp(-1j * (0 + 2 * np.pi * 1. / 3))))
    pp2 = ((irpic[2, :, :] * np.exp(-1j * (0 + 2 * np.pi * 2. / 3))))
    phases[:, :, 0] = - np.arctan2(np.real(pp0 + pp1 + pp2), np.imag(pp0 + pp1 + pp2))
    phases[:, :, 0] = phases[:, :, 0]+np.pi  #do it so as to get values from 0 to 2*pi and not -pi to pi

    #    pp3 = ((irpic[3, :, :] * np.exp(-1j * ((p0tables[:,:,1]* -0.000031 * np.pi)))))
    #    pp4 = ((irpic[4, :, :] * np.exp(-1j * (p0tables[:,:,1]* -0.000031 * np.pi + 2 * np.pi * 1. / 3))))
    #    pp5 = ((irpic[5, :, :] * np.exp(-1j * (p0tables[:,:,1]* -0.000031 * np.pi + 2 * np.pi * 2. / 3))))
    pp3 = ((irpic[3, :, :] * np.exp(-1j * (0))))
    pp4 = ((irpic[4, :, :] * np.exp(-1j * (0 + 2 * np.pi * 1. / 3))))
    pp5 = ((irpic[5, :, :] * np.exp(-1j * (0 + 2 * np.pi * 2. / 3))))
    #    print - np.angle(pp3+pp4+pp5)
    phases[:, :, 1] = - np.arctan2(np.real(pp3 + pp4 + pp5), np.imag(pp3 + pp4 + pp5))
    phases[:, :, 1] = phases[:, :, 1]+np.pi  #do it so as to get values from 0 to 2*pi and not -pi to pi

    pp6 = ((irpic[6, :, :] * np.exp(-1j * (0))))
    pp7 = ((irpic[7, :, :] * np.exp(-1j * (0 + 2 * np.pi * 1. / 3))))
    pp8 = ((irpic[8, :, :] * np.exp(-1j * (0 + 2 * np.pi * 2. / 3))))
    #    print - np.angle(pp0+pp1+pp2)
    phases[:, :, 2] = - np.arctan2(np.real(pp6 + pp7 + pp8), np.imag(pp6 + pp7 + pp8))
    phases[:, :, 2] = phases[:, :, 2]+np.pi  #do it so as to get values from 0 to 2*pi and not -pi to pi

    amps[:, :, 0] = 2. / 3. * np.absolute(pp0 + pp1 + pp2)
    amps[:, :, 1] = 2. / 3. * np.absolute(pp2 + pp3 + pp4)
    amps[:, :, 2] = 2. / 3. * np.absolute(pp5 + pp6 + pp7)
    

    cons = ({'type': 'eq', 'fun': constraint_funk})
    unwrap_coeff = np.asarray(phases, dtype=np.float32)
#    for row in range(424):
#        for col in range(512):
#            pixel_phases =phases[row,col,:]
#            res = min_ob_funk(ob_funk, pixel_phases)
##            res = minimize(ob_funk, [2., 2., 2.], constraints=cons, args=(pixel_phases,))
#            unwrap_coeff[row,col,:] = res
##            print res
#        print "Now processing: ", row, col
#    with open("unwrap_coeff.dat", 'w') as n_pha:
#        np.save(n_pha, unwrap_coeff)
        
    unwrap_coeff = np.load("unwrap_coeff.dat")
    
    plt.figure("Unwrapping Coeffs")
    plt.subplot(1, 3, 1)
    plt.imshow(unwrap_coeff[:,:,0],cmap='Greys_r')
    plt.subplot(1, 3, 2)
    plt.imshow(unwrap_coeff[:,:,1],cmap='Greys_r')
    plt.subplot(1, 3, 3)
    plt.imshow(unwrap_coeff[:,:,2],cmap='Greys_r')

    depths = np.asarray(phases,dtype=np.float32)
    
    depths = (phases + 2*np.pi*unwrap_coeff)
    
    depths[:,:,0] = depths[:,:,0] / (4*np.pi*80)  
    depths[:,:,1] = depths[:,:,1] / (4*np.pi*16)
    depths[:,:,2] = depths[:,:,2] / (4*np.pi*120)

    plt.figure("depths")
    plt.subplot(1, 3, 1)
    plt.imshow(depths[:,:,0],cmap='Greens_r')
    plt.subplot(1, 3, 2)
    plt.imshow(depths[:,:,1],cmap='Greens_r')
    plt.subplot(1, 3, 3)
    plt.imshow(depths[:,:,2],cmap='Greens_r')
    
    depth_map = np.zeros((424,512), dtype=np.float32)
    depth_map = depths.mean(axis=2)
    
    plt.figure("Depth map")
    plt.imshow(depth_map)

    scaled_phases = np.asarray(phases,dtype=np.float32)
    scaled_phases = phases + 2*np.pi*unwrap_coeff
    
    scaled_phases[:,:,0] = scaled_phases[:,:,0] / (3. * (1./80) / 2. * np.pi)
    scaled_phases[:,:,1] = scaled_phases[:,:,1] / (15. * (1./16) / 2. * np.pi)
    scaled_phases[:,:,2] = scaled_phases[:,:,2] / (2. * (1./120) / 2. * np.pi)
    
    tof = np.zeros((424,512), dtype=np.float32)
    tof= np.sum(scaled_phases, axis=2) / (1./(3. * (1./80) / 2. * np.pi) + 1./(15. * (1./16) / 2. * np.pi) + 1./(2. * (1./120) / 2. * np.pi))
    
    plt.figure("tof")
    plt.imshow(tof)    
    
    
    plt.figure("Scaled")
    plt.subplot(1, 3, 1)
    plt.imshow(scaled_phases[:,:,0],cmap='Greys_r')
    plt.subplot(1, 3, 2)
    plt.imshow(scaled_phases[:,:,1],cmap='Greys_r')
    plt.subplot(1, 3, 3)
    plt.imshow(scaled_phases[:,:,2],cmap='Greys_r')
#    T = np.asarray(phases, dtype=np.float32)
#    T[:, :, 0] = phases[:, :, 0] * 3.0 / (2. * np.pi)
#    T[:, :, 1] = phases[:, :, 1] * 15.0 / (2. * np.pi)
#    T[:, :, 2] = phases[:, :, 2] * 2.0 / (2. * np.pi)
#
#    sigma_t_0_sq = ((3. * 2. * np.pi) / (80.)) ** 2
#    sigma_t_1_sq = ((15. * 2. * np.pi) / (16.)) ** 2
#    sigma_t_2_sq = ((2. * 2. * np.pi) / (120.)) ** 2
#
#    sigma_epsilon_1_sq = sigma_t_1_sq + sigma_t_0_sq
#
#    sigma_epsilon_2_sq = sigma_t_2_sq + sigma_t_0_sq
#
#    sigma_epsilon_3_sq = sigma_t_2_sq + sigma_t_1_sq

    # %%
    ###ATTEMPT 1 to implement bilateral filter on amplitude (page 58 of the good pdf)
    amplitude_filtered = np.ndarray((424, 512, 3), dtype=np.float)
    ir_filtered = np.ndarray((424, 512), dtype=np.float)

    amplitude_filtered[:, :, 0] = np.power(-irpic[0, :, :] * np.sin(p0tables[:, :, 0]) - irpic[1, :, :] * np.sin(
        p0tables[:, :, 0] + 2 * np.pi / 3) - irpic[2, :, :] * np.sin(p0tables[:, :, 0] + 4 * np.pi / 3), 2) + np.power(
        irpic[0, :, :] * np.cos(p0tables[:, :, 0]) + irpic[1, :, :] * np.cos(p0tables[:, :, 0] + 2 * np.pi / 3) + irpic[
                                                                                                                  2, :,
                                                                                                                  :] * np.cos(
            p0tables[:, :, 0] + 4 * np.pi / 3), 2)
    amplitude_filtered[:, :, 1] = np.power(-irpic[3, :, :] * np.sin(p0tables[:, :, 1]) - irpic[4, :, :] * np.sin(
        p0tables[:, :, 1] + 2 * np.pi / 3) - irpic[5, :, :] * np.sin(p0tables[:, :, 1] + 4 * np.pi / 3), 2) + np.power(
        irpic[3, :, :] * np.cos(p0tables[:, :, 1]) + irpic[4, :, :] * np.cos(p0tables[:, :, 1] + 2 * np.pi / 3) + irpic[
                                                                                                                  5, :,
                                                                                                                  :] * np.cos(
            p0tables[:, :, 1] + 4 * np.pi / 3), 2)
    amplitude_filtered[:, :, 2] = np.power(-irpic[6, :, :] * np.sin(p0tables[:, :, 2]) - irpic[7, :, :] * np.sin(
        p0tables[:, :, 2] + 2 * np.pi / 3) - irpic[8, :, :] * np.sin(p0tables[:, :, 2] + 4 * np.pi / 3), 2) + np.power(
        irpic[6, :, :] * np.cos(p0tables[:, :, 2]) + irpic[7, :, :] * np.cos(p0tables[:, :, 2] + 2 * np.pi / 3) + irpic[
                                                                                                                  8, :,
                                                                                                                  :] * np.cos(
            p0tables[:, :, 2] + 4 * np.pi / 3), 2)

    # ir_filtered=(amplitude_filtered[:,:,0]+amplitude_filtered[:,:,1]+amplitude_filtered[:,:,2])/3
    ir_filtered = (amps[:, :, 0] + amps[:, :, 1] + amps[:, :, 2]) / 3

    # ampli_filt=plt.subplot(5,2,2)
    #    fig = plt.figure()
    #    ax1 = fig.add_subplot(221)
    #    ax1.set_title("Amplitude composed and filtered")
    # ir_filtered = exposure.equalize_hist(ir_filtered)
    # ir_filtered=np.minimum(ir_filtered,np.mean(ir_filtered))
    #    ax1.imshow(ir_filtered, cmap='Greens_r')

    # print ir_filtered


    # %%
    ## Attempt 1 at unwrapping using the idea of sawtooth minimization 
    num_points=1   # num of points that will be intersected. should be odd
    num_wraps_freq_1=10    #num of teeth the saw signal has
    num_wraps_freq_2=2
    num_wraps_freq_3=15    
    sigma=0.1              #distance between points
    
    attempt_1_depth = np.ndarray((424, 512), dtype=np.float)

    for row in range(424):
        for col in range(512):
            #intersect, the saw tih 80mhz with the phase val, retruning 10 points of intersections with 2 points of variance, of 0.1 stepsize and
            val_80 =phases[row, col, 0]
            val_16 =phases[row, col, 1]
            val_120=phases[row, col, 2]
            intersections_80 = intersect_saw  (80, num_wraps_freq_1, val_80, num_points, sigma)  
            intersections_16 = intersect_saw  (16, num_wraps_freq_2, val_16, num_points, sigma)  
            intersections_120= intersect_saw (120, num_wraps_freq_3, val_120, num_points, sigma)  
            
            
            #search through the 3 tables and find values that are closest
            #6 indices, 2 for each table
            num_rows_80 = intersections_80.shape[0]
            num_cols_80 = intersections_80.shape[1]
            
            num_rows_16 = intersections_16.shape[0]
            num_cols_16 = intersections_16.shape[1]
            
            num_rows_120 = intersections_120.shape[0]
            num_cols_120 = intersections_120.shape[1]
            
            difference=100000
            n_0=100
            n_1=100
            n_2=100
            
            mean_dist=100
            
#            print "starting the big loops"
#            for row_80 in range (num_rows_80):
#                for col_80 in range (num_cols_80):
#                    for row_16 in range (num_rows_16):
#                        for col_16 in range (num_cols_16):
#                            for row_120 in range (num_rows_120):
#                                for col_120 in range (num_cols_120):
#                                    dif=abs(abs(intersections_80[row_80,col_80]-intersections_16[row_16,col_16]) - intersections_120[row_120,col_120])
#                                    print "dif is", dif
#                                    if (dif< difference):
#                                        difference=dif
#                                        mean_dist=(intersections_80[row_80,col_80] +intersections_16[row_16,col_16]+ intersections_120[row_120,col_120])/3
##                                        
#                                    
                                    
            print "mean_dist is", mean_dist, "with difference", difference
            attempt_1_depth[row,col]=mean_dist
#            print "num_rows",num_rows_120
#            print "num_cols",num_cols_120
    # %%
    fig = plt.figure("Phases and Amplitudes")
    ccp = 'Greens'
    cca = 'Greens_r'

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove spaces between subplots

    ax = plt.subplot(4, 1, 1)
    ax.set_title("RGB image")
    ax.set_axis_off()  # Remove axis for better visualization
    plt.imshow(i)

    ax = plt.subplot(4, 3, 4)
    ax.set_axis_off()  # Remove axis for better visualization
    # ax.set_title("Frequency 1 phase-shift")
    plt.imshow((phases[:, :, 0]), cmap=ccp)

    ax = plt.subplot(4, 3, 5)
    ax.set_axis_off()
    # ax.set_title("Frequency 2 phase-shift")
    plt.imshow((phases[:, :, 1]), cmap=ccp)

    ax = plt.subplot(4, 3, 6)
    ax.set_axis_off()
    # ax.set_title("Frequency 3 phase-shift")
    plt.imshow((phases[:, :, 2]), cmap=ccp)

    ax = plt.subplot(4, 3, 7)
    ax.set_axis_off()
    # ax.set_title("Frequency 1 Amplitude")
    plt.imshow(amps[:, :, 0], cmap=cca)

    ax = plt.subplot(4, 3, 8)
    ax.set_axis_off()
    # ax.set_title("Frequency 2 Amplitude")
    plt.imshow(amps[:, :, 1], cmap=cca)

    ax = plt.subplot(4, 3, 9)
    ax.set_axis_off()
    # ax.set_title("Frequency 3 Amplitude")
    plt.imshow(amps[:, :, 2], cmap=cca)

    phases[:, :, 0] = phases[:, :, 0]
    phases[:, :, 1] = phases[:, :, 1]
    phases[:, :, 2] = phases[:, :, 2]

    # Showing the unrwapped ones
    phases_unwrapped = np.ndarray((424, 512, 3), dtype=np.float)
    phases_unwrapped[:, :, 0] = unwrap_phase(phases[:, :, 0])
    phases_unwrapped[:, :, 1] = unwrap_phase(phases[:, :, 1])
    phases_unwrapped[:, :, 2] = unwrap_phase(phases[:, :, 2])

    print np.mean(phases[:, :, 0]), np.max(phases[:, :, 0]), np.min(phases[:, :, 0])
    print np.mean(phases_unwrapped[:, :, 0]), np.max(phases_unwrapped[:, :, 0]), np.min(phases_unwrapped[:, :, 0])

    ax = plt.subplot(4, 3, 10)
    ax.set_axis_off()
    # ax.set_title("Composite amplitudes")
    plt.imshow(phases_unwrapped[:, :, 0], cmap=ccp)

    ax = plt.subplot(4, 3, 11)
    ax.set_axis_off()
    # ax.set_title("Composite amplitudes")
    plt.imshow(phases_unwrapped[:, :, 1], cmap=ccp)

    ax = plt.subplot(4, 3, 12)
    ax.set_axis_off()
    # ax.set_title("Composite amplitudes")
    plt.imshow(phases_unwrapped[:, :, 2], cmap=ccp)

    print "phases.shape:", phases.shape

    unwrapped_ = unwrap_superior(medfilt(phases + np.pi))
    plt.figure("superior")
    plt.imshow((unwrapped_), cmap=ccp)
    #    plt.subplot(1, 3, 1)
    #    plt.imshow((unwrapped_[:,:,0]), cmap=ccp)
    #    plt.subplot(1, 3, 2)
    #    plt.imshow((unwrapped_[:,:,1]), cmap=ccp)
    #    plt.subplot(1, 3, 3)
    #    plt.imshow((unwrapped_[:,:,2]), cmap=ccp)
    print "unwrapped: ", unwrapped_.shape, unwrapped_.max(), unwrapped_.min()
    with open("unwrapped.dat", 'w') as unw:
        np.save(unw, unwrapped_)

    plt.show()
    print "finished"

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
from testing_sawtooth import *

from skimage.restoration import unwrap_phase
from skimage import exposure
from skimage import filter
from scipy.signal import medfilt
from scipy.optimize import minimize
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import cv2


ccp = 'Greens_r'
cca = 'Greens_r'

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
#    with open("../data/20160716-133438864.ir_a1", "rb") as ir_a1:
#        a1__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_a1")
#        plt.imshow(a1__) 
#        
#    with open("../data/20160716-133438864.ir_b1", "rb") as ir_a1:
#        b1__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_b1")
#        plt.imshow(b1__) 
#
#    with open("../data/20160716-133438864.ir_amp1", "rb") as ir_a1:
#        amp1__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_amp1")
#        plt.imshow(amp1__) 
#    
#    
#    with open("../data/20160716-133438864.ir_a2", "rb") as ir_a1:
#        a2__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_a2")
#        plt.imshow(a2__) 
#        
#    with open("../data/20160716-133438864.ir_b2", "rb") as ir_a1:
#        b2__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_b2")
#        plt.imshow(b2__) 
#
#    with open("../data/20160716-133438864.ir_amp2", "rb") as ir_a1:
#        amp2__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_amp2")
#        plt.imshow(amp2__) 
#    
#    
#    with open("../data/20160716-133438864.ir_a3", "rb") as ir_a1:
#        a3__ =np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_a3")
#        plt.imshow(a3__) 
#        
#    with open("../data/20160716-133438864.ir_b3", "rb") as ir_a1:
#        b3__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_b3")
#        plt.imshow(b3__) 
#
#    with open("../data/20160716-133438864.ir_amp3", "rb") as ir_a1:
#        amp3__ = np.flipud(np.fromfile(ir_a1, dtype=np.float32).reshape((424, 512)))
#        plt.figure("ir_amp3")
#        plt.imshow(amp3__)     
##    
    with open("../data/20160716-182441518.ir_phase0", "rb") as ir_a1_F:
        a3__ =np.flipud(np.fromfile(ir_a1_F, dtype=np.float32).reshape((424, 512)))
        plt.figure("ir_phase0")
        plt.imshow(a3__,cmap='Greys_r')
        
    with open("../data/20160716-182441518.ir_phase1", "rb") as ir_a1_F:
        b3__ = np.flipud(np.fromfile(ir_a1_F, dtype=np.float32).reshape((424, 512)))
        plt.figure("ir_phase1")
        plt.imshow(b3__,cmap='Greys_r')

    with open("../data/20160716-182441518.ir_phase2", "rb") as ir_a1_F:
        amp3__ = np.flipud(np.fromfile(ir_a1_F, dtype=np.float32).reshape((424, 512)))
        plt.figure("ir_phase2")
        plt.imshow(amp3__,cmap='Greys_r') 
    
    np.savetxt('ir_phase0.out',a3__, delimiter=',',fmt='%f')
    np.savetxt('ir_phase1.out',b3__, delimiter=',',fmt='%f')
    np.savetxt('ir_phase2.out',amp3__, delimiter=',',fmt='%f')
    STOP
    
    with open("p0tables.dat", "rb") as p0tf:
        p0tables = np.fromfile(p0tf, dtype=np.uint16)
        p0tables = np.flipud(p0tables.reshape((3, 424, 512)))
        print p0tables.shape
        plt.figure("p0 Tables")
        plt.subplot(1, 3, 1)
        plt.imshow(p0tables[0,:,:])
        plt.subplot(1, 3, 2)
        plt.imshow(p0tables[1,:,:])
        plt.subplot(1, 3, 3)
        plt.imshow(p0tables[2,:,:])
        
#        np.savetxt('p0_0.out',p0tables[0,:,:], delimiter=',',fmt='%d')
#        np.savetxt('p0_1.out',p0tables[1,:,:], delimiter=',',fmt='%d')
#        np.savetxt('p0_2.out',p0tables[2,:,:], delimiter=',',fmt='%d')
        

    # with open("decodedIR", "rb") as f2:
    with open("../data/20160716-142209598.ir", "rb") as f2: # Alex_room
#    with open("../data/20160714-190318193.ir", "rb") as f2: # Corridor
#    with open("../data/20160714-190254793.ir", "rb") as f2: # Wall
#    with open("../data/20160616-120946344.ir", "rb") as f2: # Boris
#    with open("../data/20160616-12434139.ir", "rb") as f2: #Corbin
        plt.figure("Raw data")        
        irpic = np.fromfile(f2, dtype=np.int32)
        print (irpic.shape)
        irpic = (np.minimum(abs(((irpic.reshape((9, 424, 512))))),500))
        
        irpic = irpic.astype(dtype=np.float32)
        for pp in range(9):
            irpic[pp,:,:] = cv2.bilateralFilter(irpic[pp,:,:],7, 40, 10)
#            irpic[pp,:,:] = denoise_bilateral(irpic[pp,:,:], win_size=7, sigma_color=40, sigma_spatial=10, multichannel=False)

        print (irpic.shape)

        ax = plt.subplot(4, 1, 1)

        #       i = imread("dumprgb12.jpeg")
        i = imread("../data/20160716-142209586.rgb") #Alex_room
#        i = imread("../data/20160714-190318147.rgb") #Corridor
#        i = imread("../data/20160714-190254747.rgb") #Wall
#        i = imread("../data/20160616-120946332.rgb") #Boris
#        i = imread("../data/20160616-12434225.rgb") #Corbin

        ax.set_axis_off()

        plt.imshow(i)
        counter = 4
        for row in range(2, 5):
            for col in range(2, 5):
                # flip once
                irpic[counter - 4][:][:] = np.flipud(irpic[counter - 4][:][:])
                ax = plt.subplot(4, 3, counter)
                ax.set_axis_off()
#                plt.imshow(np.minimum(irpic[counter - 4][:][:], np.mean(irpic[counter - 4][:][:])), cmap='inferno')
                plt.imshow(irpic[counter - 4][:][:], cmap='inferno')
                counter = counter + 1
                
#        np.savetxt('ir_0.out',irpic[0,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_1.out',irpic[1,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_2.out',irpic[2,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_3.out',irpic[3,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_4.out',irpic[4,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_5.out',irpic[5,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_6.out',irpic[6,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_7.out',irpic[7,:,:], delimiter=',',fmt='%d')
#        np.savetxt('ir_8.out',irpic[8,:,:], delimiter=',',fmt='%d')
#        
#        exit()
        
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

    # %% COSINES AND SINES METHOD
    proc_p0tables = np.asarray(p0tables, dtype=np.float32) * -0.000031 * np.pi
    tmp0 = proc_p0tables + 0
    tmp1 = proc_p0tables + 2. * np.pi / 3.
    tmp2 = proc_p0tables + 4. * np.pi / 3.
    
    
    new_p0_0 = np.zeros((6,424,512), dtype=np.float32)
    new_p0_1 = np.zeros((6,424,512), dtype=np.float32)
    new_p0_2 = np.zeros((6,424,512), dtype=np.float32)

    
    new_p0_0[0,:,:] = np.cos(tmp0[0,:,:] )
    new_p0_0[1,:,:] = np.cos(tmp1[0,:,:])
    new_p0_0[2,:,:] = np.cos(tmp2[0,:,:])
    
    new_p0_0[3,:,:] = np.sin(-tmp0[0,:,:])
    new_p0_0[4,:,:] = np.sin(-tmp1[0,:,:])
    new_p0_0[5,:,:] = np.sin(-tmp2[0,:,:])

 
    new_p0_1[0,:,:] = np.cos(tmp0[1,:,:])
    new_p0_1[1,:,:] = np.cos(tmp1[1,:,:])
    new_p0_1[2,:,:] = np.cos(tmp2[1,:,:])
    
    new_p0_1[3,:,:] = np.sin(-tmp0[1,:,:])
    new_p0_1[4,:,:] = np.sin(-tmp1[1,:,:])
    new_p0_1[5,:,:] = np.sin(-tmp2[1,:,:])

 
    new_p0_2[0,:,:] = np.cos(tmp0[2,:,:])
    new_p0_2[1,:,:] = np.cos(tmp1[2,:,:])
    new_p0_2[2,:,:] = np.cos(tmp2[2,:,:])
    
    new_p0_2[3,:,:] = np.sin(-tmp0[2,:,:])
    new_p0_2[4,:,:] = np.sin(-tmp1[2,:,:])
    new_p0_2[5,:,:] = np.sin(-tmp2[2,:,:])


    for row in range(424):
        for col in range(512):
            ir_a = irpic[0,row,col] * new_p0_0[0,row,col] + irpic[1,row,col] * new_p0_0[1,row,col] + irpic[2,row,col] * new_p0_0[2,row,col]
            ir_b = irpic[0,row,col] * new_p0_0[3,row,col] + irpic[1,row,col] * new_p0_0[4,row,col] + irpic[2,row,col] * new_p0_0[5,row,col]
            

            ir_a = ir_a * 1.322581
            ir_b = ir_b * 1.322581
            
            amps[row,col,0] = np.sqrt(ir_a**2 + ir_b**2)
            tmp = np.arctan2(ir_b, ir_a)
            if tmp < 0:
                tmp = tmp + 2. *np.pi
            phases[row,col,0] = tmp
            
            ir_a = irpic[3,row,col] * new_p0_1[0,row,col] + irpic[4,row,col] * new_p0_1[1,row,col] + irpic[5,row,col] * new_p0_1[2,row,col]
            ir_b = irpic[3,row,col] * new_p0_1[3,row,col] + irpic[4,row,col] * new_p0_1[4,row,col] + irpic[5,row,col] * new_p0_1[5,row,col]
            
            ir_a = ir_a * 1.
            ir_b = ir_b * 1.
            
            amps[row,col,1] = np.sqrt(ir_a**2 + ir_b**2)
            tmp = np.arctan2(ir_b, ir_a)
            if tmp < 0:
                tmp = tmp + 2. *np.pi
            phases[row,col,1] = tmp
            
            ir_a = irpic[6,row,col] * new_p0_2[0,row,col] + irpic[7,row,col] * new_p0_2[1,row,col] + irpic[8,row,col] * new_p0_2[2,row,col]
            ir_b = irpic[6,row,col] * new_p0_2[3,row,col] + irpic[7,row,col] * new_p0_2[4,row,col] + irpic[8,row,col] * new_p0_2[5,row,col]
            
            ir_a = ir_a * 1.612903
            ir_b = ir_b * 1.612903
            
            amps[row,col,2] = np.sqrt(ir_a**2 + ir_b**2)
            tmp = np.arctan2(ir_b, ir_a)
            if tmp < 0:
                tmp = tmp + 2. *np.pi
            phases[row,col,2] = tmp

#    plt.figure("new apppssss")
#    plt.subplot(1, 3, 1)
#    plt.imshow(amps[:,:,0])
#    plt.subplot(1, 3, 2)
#    plt.imshow(amps[:,:,1])
#    plt.subplot(1, 3, 3)
#    plt.imshow(amps[:,:,2])
#    
#    plt.figure("new phaaaases")
#    plt.subplot(1, 3, 1)
#    plt.imshow(phases[:,:,0])
#    plt.subplot(1, 3, 2)
#    plt.imshow(phases[:,:,1])
#    plt.subplot(1, 3, 3)
#    plt.imshow(phases[:,:,2])
#    
#    aps_ave = np.mean(amps,axis=2)
#    plt.figure("new amps ave")
#    plt.imshow(aps_ave)
#    plt.show()




#%%
    proc_p0tables = np.flipud(p0tables) * -0.000031 * np.pi
    proc_p0tables[1,:, :] = proc_p0tables[1,:, :] + 2. * np.pi / 3.
    proc_p0tables[2,:, :] = proc_p0tables[2,:, :] + 4. * np.pi / 3.

    cos_p0tables = np.cos(proc_p0tables)
    sin_p0tables = np.sin(-1. * proc_p0tables)

    #    m[0] == 32767 || m[1] == 32767 || m[2] == 32767



    ir_a1 = cos_p0tables[0,:, :] * irpic[0, :, :] + cos_p0tables[1,:, :] * irpic[1, :, :] + cos_p0tables[2,:, :] * irpic[2, :, :]
    ir_b1 = sin_p0tables[0,:, :] * irpic[0, :, :] + sin_p0tables[1,:, :] * irpic[1, :, :] + sin_p0tables[2,:, :] * irpic[2, :, :]

    ir_a1 = ir_a1 * 1.322581
    ir_b1 = ir_b1 * 1.322581

#    amps[:, :, 0] = np.absolute(1j*ir_a1 , ir_b1)  #np.sqrt(ir_a1 ** 2 + ir_b1 ** 2)
#    phases[:, :, 0] = np.arctan2(ir_b1, ir_a1)


    ir_a1[irpic[0, :, :] == 32767] = 0.
    ir_a1[irpic[1, :, :] == 32767] = 0.
    ir_a1[irpic[2, :, :] == 32767] = 0.

    ir_b1[irpic[0] == 32767] = 0.
    ir_b1[irpic[1] == 32767] = 0.
    ir_b1[irpic[2] == 32767] = 0.

    amps[:, :, 0][irpic[0] == 32767] = 0.
    amps[:, :, 0][irpic[1] == 32767] = 0.
    amps[:, :, 0][irpic[2] == 32767] = 0.

    ir_a2 = cos_p0tables[0,:, :] * irpic[3, :, :] + cos_p0tables[1,:, :] * irpic[4, :, :] + cos_p0tables[2,:, :] * irpic[5, :, :]
    ir_b2 = sin_p0tables[0,:, :] * irpic[3, :, :] + sin_p0tables[1,:, :] * irpic[4, :, :] + sin_p0tables[2,:, :] * irpic[5, :, :]

    ir_a2 = ir_a2 * 1.
    ir_b2 = ir_b2 * 1.

#    amps[:, :, 1] = np.sqrt(ir_a2 ** 2 + ir_b2 ** 2)
    #    phases[:, :, 1] = np.arctan2(ir_b2, ir_a2)



    ir_a2[irpic[3, :, :] == 32767] = 0.
    ir_a2[irpic[4, :, :] == 32767] = 0.
    ir_a2[irpic[5, :, :] == 32767] = 0.

    ir_b2[irpic[3] == 32767] = 0.
    ir_b2[irpic[4] == 32767] = 0.
    ir_b2[irpic[5] == 32767] = 0.

    amps[:, :, 1][irpic[3] == 32767] = 0.
    amps[:, :, 1][irpic[4] == 32767] = 0.
    amps[:, :, 1][irpic[5] == 32767] = 0.

    ir_a3 = cos_p0tables[0,:, :] * irpic[6, :, :] + cos_p0tables[1,:, :] * irpic[7, :, :] + cos_p0tables[2,:, :] * irpic[8, :, :]
    ir_b3 = sin_p0tables[0,:, :] * irpic[6, :, :] + sin_p0tables[1,:, :] * irpic[7, :, :] + sin_p0tables[2,:, :] * irpic[8, :, :]

    ir_a3 = ir_a3 * 1.612903
    ir_b3 = ir_b3 * 1.612903

#    amps[:, :, 2] = np.sqrt(ir_a3 ** 2 + ir_b3 ** 2)
    #    phases[:, :, 2] = np.arctan2(np.real(ir_b3), np.imag(ir_a3))


    ir_a3[irpic[6, :, :] == 32767] = 0.
    ir_a3[irpic[7, :, :] == 32767] = 0.
    ir_a3[irpic[8, :, :] == 32767] = 0.

    ir_b3[irpic[6] == 32767] = 0.
    ir_b3[irpic[7] == 32767] = 0.
    ir_b3[irpic[8] == 32767] = 0.

    amps[:, :, 2][irpic[6] == 32767] = 0.
    amps[:, :, 2][irpic[7] == 32767] = 0.
    amps[:, :, 2][irpic[8] == 32767] = 0.

    ir_amplitude = amps.mean(axis=2)

    ir_p = - np.arctan(amps.sum(axis=2))

    #    plt.figure("ir_a")
    #    plt.imshow(ir_a)
    #    plt.figure("ir_b")
    #    plt.imshow(ir_b)
    amps_eq = np.minimum(ir_amplitude, 1000)

    #    plt.figure("ir_amp")
    #    plt.imshow(amps_eq,cmap="Greens_r")
    #    plt.figure("ir_pha")
    #    plt.imshow(ir_p)




    # %% EXPONENTIAION METHOD
    pp0 = ((irpic[0, :, :] * np.exp(-1j * (0))))
    pp1 = ((irpic[1, :, :] * np.exp(-1j * (0 + 2 * np.pi * 1. / 3))))
    pp2 = ((irpic[2, :, :] * np.exp(-1j * (0 + 2 * np.pi * 2. / 3))))
    phases[:, :, 0] = - np.arctan2(np.real(pp0 + pp1 + pp2), np.imag(pp0 + pp1 + pp2))
    #    pp3 = ((irpic[3, :, :] * np.exp(-1j * ((p0tables[:,:,1]* -0.000031 * np.pi)))))
    #    pp4 = ((irpic[4, :, :] * np.exp(-1j * (p0tables[:,:,1]* -0.000031 * np.pi + 2 * np.pi * 1. / 3))))
    #    pp5 = ((irpic[5, :, :] * np.exp(-1j * (p0tables[:,:,1]* -0.000031 * np.pi + 2 * np.pi * 2. / 3))))
    pp3 = ((irpic[3, :, :] * np.exp(-1j * (0))))
    pp4 = ((irpic[4, :, :] * np.exp(-1j * (0 + 2 * np.pi * 1. / 3))))
    pp5 = ((irpic[5, :, :] * np.exp(-1j * (0 + 2 * np.pi * 2. / 3))))
    #    print - np.angle(pp3+pp4+pp5)
    phases[:, :, 1] = - np.arctan2(np.real(pp3 + pp4 + pp5), np.imag(pp3 + pp4 + pp5))

    pp6 = ((irpic[6, :, :] * np.exp(-1j * (0))))
    pp7 = ((irpic[7, :, :] * np.exp(-1j * (0 + 2 * np.pi * 1. / 3))))
    pp8 = ((irpic[8, :, :] * np.exp(-1j * (0 + 2 * np.pi * 2. / 3))))
    #    print - np.angle(pp0+pp1+pp2)
    phases[:, :, 2] = - np.arctan2(np.real(pp6 + pp7 + pp8), np.imag(pp6 + pp7 + pp8))

    phases[phases<0] = phases[phases<0] + 2. * np.pi
#    amps[:, :, 0] = 2. / 3. * np.absolute(pp0 + pp1 + pp2)
#    amps[:, :, 1] = 2. / 3. * np.absolute(pp2 + pp3 + pp4)
#    amps[:, :, 2] = 2. / 3. * np.absolute(pp5 + pp6 + pp7)
    #%%

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
        
#    unwrap_coeff = np.load("unwrap_coeff.dat")
    
#    plt.figure("Unwrapping Coeffs")
#    plt.subplot(1, 3, 1)
#    plt.imshow(unwrap_coeff[:,:,0],cmap='Greens_r')
#    plt.subplot(1, 3, 2)
#    plt.imshow(unwrap_coeff[:,:,1],cmap='Greens_r')
#    plt.subplot(1, 3, 3)
#    plt.imshow(unwrap_coeff[:,:,2],cmap='Greens_r')

    depths = np.asarray(phases,dtype=np.float32)
    
    depths = (phases + 2*np.pi*unwrap_coeff)
    
    depths[:,:,0] = depths[:,:,0] / (4*np.pi*80)  
    depths[:,:,1] = depths[:,:,1] / (4*np.pi*16)
    depths[:,:,2] = depths[:,:,2] / (4*np.pi*120)

#    plt.figure("depths")
#    plt.subplot(1, 3, 1)
#    plt.imshow(depths[:,:,0],cmap='Greens_r')
#    plt.subplot(1, 3, 2)
#    plt.imshow(depths[:,:,1],cmap='Greens_r')
#    plt.subplot(1, 3, 3)
#    plt.imshow(depths[:,:,2],cmap='Greens_r')
    
    depth_map = np.zeros((424,512), dtype=np.float32)
    depth_map = depths.mean(axis=2)
    
#    plt.figure("Depth map")
#    plt.imshow(depth_map)

    scaled_phases = np.asarray(phases,dtype=np.float32)
    scaled_phases = phases + 2*np.pi*unwrap_coeff
    
    scaled_phases[:,:,0] = scaled_phases[:,:,0] / (3. * (1./80) / 2. * np.pi)
    scaled_phases[:,:,1] = scaled_phases[:,:,1] / (15. * (1./16) / 2. * np.pi)
    scaled_phases[:,:,2] = scaled_phases[:,:,2] / (2. * (1./120) / 2. * np.pi)
    
    tof = np.zeros((424,512), dtype=np.float32)
    tof= np.sum(scaled_phases, axis=2) / (1./(3. * (1./80) / 2. * np.pi) + 1./(15. * (1./16) / 2. * np.pi) + 1./(2. * (1./120) / 2. * np.pi))
    
#    plt.figure("tof")
#    plt.imshow(tof)    
    
    
#    plt.figure("Scaled")
#    plt.subplot(1, 3, 1)
#    plt.imshow(scaled_phases[:,:,0],cmap='Greens_r')
#    plt.subplot(1, 3, 2)
#    plt.imshow(scaled_phases[:,:,1],cmap='Greens_r')
#    plt.subplot(1, 3, 3)
#    plt.imshow(scaled_phases[:,:,2],cmap='Greens_r')
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

    amplitude_filtered[:, :, 0] = np.power(-irpic[0, :, :] * np.sin(p0tables[0,:, :]) - irpic[1, :, :] * np.sin(
        p0tables[0,:, :] + 2 * np.pi / 3) - irpic[2, :, :] * np.sin(p0tables[0,:, :] + 4 * np.pi / 3), 2) + np.power(
        irpic[0, :, :] * np.cos(p0tables[0,:, :]) + irpic[1, :, :] * np.cos(p0tables[0,:, :] + 2 * np.pi / 3) + irpic[
                                                                                                                  2, :,
                                                                                                                  :] * np.cos(
            p0tables[0,:, :] + 4 * np.pi / 3), 2)
    amplitude_filtered[:, :, 1] = np.power(-irpic[3, :, :] * np.sin(p0tables[1,:, :]) - irpic[4, :, :] * np.sin(
        p0tables[1,:, :] + 2 * np.pi / 3) - irpic[5, :, :] * np.sin(p0tables[1,:, :] + 4 * np.pi / 3), 2) + np.power(
        irpic[3, :, :] * np.cos(p0tables[1,:, :]) + irpic[4, :, :] * np.cos(p0tables[1,:, :] + 2 * np.pi / 3) + irpic[
                                                                                                                  5, :,
                                                                                                                  :] * np.cos(
            p0tables[1,:, :] + 4 * np.pi / 3), 2)
    amplitude_filtered[:, :, 2] = np.power(-irpic[6, :, :] * np.sin(p0tables[2,:, :]) - irpic[7, :, :] * np.sin(
        p0tables[2,:, :] + 2 * np.pi / 3) - irpic[8, :, :] * np.sin(p0tables[2,:, :] + 4 * np.pi / 3), 2) + np.power(
        irpic[6, :, :] * np.cos(p0tables[2,:, :]) + irpic[7, :, :] * np.cos(p0tables[2,:, :] + 2 * np.pi / 3) + irpic[
                                                                                                                  8, :,
                                                                                                                  :] * np.cos(
            p0tables[2,:, :] + 4 * np.pi / 3), 2)

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

#    phases[:, :, 0] = np.arctan2(b1__, a1__)
#    phases[:, :, 1] = np.arctan2(b2__, a2__)
#    phases[:, :, 2] = np.arctan2(b3__, a3__)
    phases[phases<0] = phases[phases<0] + 2.*np.pi



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

    unwrapped_, confidence_map = unwrap_superior(phases)
    plt.figure("superior")
    plt.imshow((unwrapped_), cmap=ccp)

    plt.figure("confidence_map")
    plt.imshow((confidence_map), cmap=ccp)

    print "unwrapped: ", unwrapped_.shape, unwrapped_.max(), unwrapped_.min()
    with open("unwrapped.dat", 'w') as unw:
        np.save(unw, unwrapped_)

    plt.show()
    print "finished"

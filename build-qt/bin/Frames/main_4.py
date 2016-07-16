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


def get_trig_tables(p0tables_):
    proc_p0tables_ = np.asarray(p0tables_, dtype=np.float32) * -0.000031 * np.pi
    tmp0 = proc_p0tables_ + 0.
    tmp1 = proc_p0tables_ + 2. * np.pi / 3.
    tmp2 = proc_p0tables_ + 4. * np.pi / 3.
    
    
    trig_tab = np.zeros((6,424,512), dtype=np.float32)
    
    trig_tab[0] = np.cos(tmp0)
    trig_tab[1] = np.cos(tmp1)
    trig_tab[2] = np.cos(tmp2)
    
    trig_tab[3] = np.sin(-tmp0)
    trig_tab[4] = np.sin(-tmp1)
    trig_tab[5] = np.sin(-tmp2)
    return trig_tab


def process_measurement_triple(trig_table0, ab_multiplier, raw_volts):
    cos_tmp0 = trig_table0[0]
    cos_tmp1 = trig_table0[1]
    cos_tmp2 = trig_table0[2]
    
    sin_negtmp0 = trig_table0[3]
    sin_negtmp1 = trig_table0[4]
    sin_negtmp2 = trig_table0[5]

    ir_a = cos_tmp0 * raw_volts[0] + cos_tmp1 * raw_volts[1] + cos_tmp2 * raw_volts[2]
    ir_b = sin_negtmp0 * raw_volts[0] + sin_negtmp1 * raw_volts[1] + sin_negtmp2 * raw_volts[2]
    
    ir_a *= ab_multiplier    
    ir_b *= ab_multiplier    
    
    ir_amp = np.sqrt(np.square(ir_a) + np.square(ir_b)) * 2./3.
    
    saturated_ind =  np.where(raw_volts[0] == 32767)
    ir_a[saturated_ind] = 0.
    ir_b[saturated_ind] = 0.
    ir_amp[saturated_ind] = 65535.0
    
    saturated_ind =  np.where(raw_volts[1] == 32767)
    ir_a[saturated_ind] = 0.
    ir_b[saturated_ind] = 0.
    ir_amp[saturated_ind] = 65535.0
    
    saturated_ind =  np.where(raw_volts[2] == 32767)
    ir_a[saturated_ind] = 0.
    ir_b[saturated_ind] = 0.
    ir_amp[saturated_ind] = 65535.0
    
    return ir_a, ir_b, ir_amp
    
def transform_measurements(a, b, ab_multiplier):
    #phase    
    tmp0 = np.arctan2(b, a)
    tmp0[tmp0 < 0] += 2. * np.pi 
    phase = np.nan_to_num(tmp0)

    #ir amplitude
    amp = np.sqrt(np.square(a) + np.square(b)) * ab_multiplier;

    return phase, amp
    
if __name__ == "__main__":
    print "started"
    
    phases = np.ndarray((3, 424, 512), dtype=np.float)
    amps = np.ndarray((3, 424, 512), dtype=np.float)

    with open("p0tables.dat", "rb") as p0tf:
        p0tables = np.fromfile(p0tf, dtype=np.uint16)
        p0tables = p0tables.reshape((3, 424, 512))
    #TODO: check that the p0tables are right

    with open("../data/20160716-142209598.ir", "rb") as f2:
        irpic = np.fromfile(f2, dtype=np.int32).reshape((9, 424, 512))
     
    irpic = irpic.astype(dtype=np.float32)
    for i in range(9):
        irpic[i] = np.flipud(irpic[i])
#        irpic[i] = cv2.bilateralFilter(irpic[i],7, 40, 10)
        
#    np.savetxt('ir_0.out',irpic[0,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_1.out',irpic[1,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_2.out',irpic[2,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_3.out',irpic[3,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_4.out',irpic[4,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_5.out',irpic[5,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_6.out',irpic[6,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_7.out',irpic[7,:,:], delimiter=',',fmt='%d')
#    np.savetxt('ir_8.out',irpic[8,:,:], delimiter=',',fmt='%d')
    
    
#    ax = plt.subplot(4, 1, 1)
#
#    ax.set_axis_off()
#
#    counter = 4
#    for row in range(2, 5):
#        for col in range(2, 5):
#            # flip once
#            irpic[counter - 4][:][:] = irpic[counter - 4][:][:]
#            ax = plt.subplot(4, 3, counter)
#            ax.set_axis_off()
##                plt.imshow(np.minimum(irpic[counter - 4][:][:], np.mean(irpic[counter - 4][:][:])), cmap='inferno')
#            plt.imshow(np.minimum(irpic[counter - 4][:][:],500), cmap='inferno')
#            counter = counter + 1

    
    trig_table0 = get_trig_tables(p0tables[0])
    trig_table1 = get_trig_tables(p0tables[1])
    trig_table2 = get_trig_tables(p0tables[2])
    
    ab_multiplier0 =  1.322581
    ab_multiplier1 =  1.
    ab_multiplier2 =  1.612903
    
    ir_a1, ir_b1, ir_amp1 = process_measurement_triple(trig_table0, ab_multiplier0, irpic[0:3])
    ir_a2, ir_b2, ir_amp2 = process_measurement_triple(trig_table1, ab_multiplier1, irpic[3:6])
    ir_a3, ir_b3, ir_amp3 = process_measurement_triple(trig_table2, ab_multiplier2, irpic[6:9])
    
    phase0, amp0 = transform_measurements(ir_a1, ir_b1, ab_multiplier0)
    phase1, amp1 = transform_measurements(ir_a2, ir_b2, ab_multiplier1)
    phase2, amp2 = transform_measurements(ir_a3, ir_b3, ab_multiplier2)
    
    
    ab_output_multiplier = 16.0
    ir_out = np.minimum((amp0 + amp1 + amp2 / 3.0) * ab_output_multiplier, 65535.0)

    
    plt.figure("ir_out")
    plt.imshow(ir_out, cmap=ccp)
    plt.colorbar()    

    plt.figure("phase0")
    plt.imshow((phase0), cmap=ccp)
    plt.colorbar()    
    plt.figure("phase1")
    plt.imshow((phase1), cmap=ccp)
    plt.colorbar()    
    plt.figure("phase2")
    plt.imshow((phase2), cmap=ccp)
    plt.colorbar()    


     
    # Showing the unrwapped ones
#    phases_unwrapped = np.ndarray((3, 424, 512), dtype=np.float)
#    phases_unwrapped[0] = unwrap_phase(phase0)
#    phases_unwrapped[1] = unwrap_phase(phase1)
#    phases_unwrapped[2] = unwrap_phase(phase2)
    
#    plt.figure("Numpy unwrapped")
#    ax = plt.subplot(1, 3, 1)
#    ax.set_axis_off()
#    plt.imshow(phases_unwrapped[0, :, :], cmap=ccp)

#    ax = plt.subplot(1, 3, 2)
#    ax.set_axis_off()
#    plt.imshow(phases_unwrapped[1, :, :], cmap=ccp)
#
#    ax = plt.subplot(1, 3, 3)
#    ax.set_axis_off()
#    plt.imshow(phases_unwrapped[2, :, :], cmap=ccp)
    
    unwrapped_, confidence_map = unwrap_superior_newidx(np.array([phase0,phase1,phase2]))
    unwrapped_ = unwrapped_ * 18.5 / 10000.
    confidence_map = confidence_map * 18.5 / 10000.
    plt.figure("superior")
    plt.imshow((unwrapped_), cmap=ccp)
    plt.colorbar()
    
    plt.figure("confidence_map")
    plt.imshow((confidence_map), cmap=ccp)
    plt.colorbar()


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
from math import atan2, sin, cos



if __name__ == "__main__" :
    print "started"

#    f = open("/home/ahmad/Documents/Masters/Computational-Photography/project/libfreenect2/build-qt/bin/Frames/tab11to16.dat", "r")
#    a = np.fromfile(f, dtype=np.uint16)
#    print a

    with open("p0tables.dat", "rb") as p0tf:
        p0tables = np.fromfile(p0tf, dtype=np.uint16)
        p0tables = np.flipud(p0tables.reshape((424,512,3)))
        print p0tables.shape

    with open("decodedIR", "rb") as f2:
       plt.figure("Raw data")
       irpic = np.fromfile(f2, dtype=np.int32)
       print (irpic.shape)
       irpic = (abs(((irpic.reshape((9,424,512))))))
       print (irpic.shape)

       plt.subplot(4,1,1)

       i = imread("dumprgb12.jpeg")


       plt.imshow(i)
       counter = 4
       for row in range(2,5):
           for col in range(2,5):
               # flip once 
               irpic[counter-4][:][:] = np.flipud(irpic[counter-4][:][:])
               plt.subplot(4, 3, counter)
               plt.imshow(np.minimum(irpic[counter-4][:][:],np.mean(irpic[counter-4][:][:])))
               counter = counter +1

   
    
    #%%    
    ### ACCORDING TO PATENT
    phases = np.ndarray((424,512,3),dtype=np.float)
    amps   = np.ndarray((424,512,3),dtype=np.float)

    
    first_term = irpic[0,:,:] * np.sin (0) + irpic[1,:,:] * np.sin (2*np.pi/3) + irpic[2,:,:] * np.sin(4*np.pi/3) 
    second_term = irpic[0,:,:] * np.cos(0) + irpic[1,:,:] * np.cos(2*np.pi/3) +  irpic[2,:,:] * np.cos(4*np.pi/3) 
    
    phases[:,:,0]=np.arctan2(-first_term,second_term)
    amps[:,:,0]=np.sqrt(np.square(second_term)+np.square(first_term))/2.
    
    first_term = irpic[3,:,:] * sin (0) + irpic[4,:,:] * sin (2*np.pi/3) + irpic[5,:,:] * sin(4*np.pi/3) 
    second_term = irpic[3,:,:] * cos(0) + irpic[4,:,:] * cos(2*np.pi/3) +  irpic[5,:,:] * cos(4*np.pi/3) 
    
    phases[:,:,1]=np.arctan2(-(first_term),(second_term))
    amps[:,:,1]=np.sqrt(np.square(second_term)+np.square(first_term))/2.
    
    first_term = irpic[6,:,:] * sin (0) + irpic[7,:,:] * sin (2*np.pi/3) + irpic[8,:,:] * sin(4*np.pi/3) 
    second_term = irpic[6,:,:] * cos(0) + irpic[7,:,:] * cos(2*np.pi/3) +  irpic[8,:,:] * cos(4*np.pi/3) 
    
    phases[:,:,2]=np.arctan2(-(first_term),(second_term))
    amps[:,:,2]=np.sqrt(np.square(second_term)+np.square(first_term))/2.
    ### END ACCORDING TO PATENT

#%%

    pp0 = ((irpic[0,:,:]*np.exp(-1j*(p0tables[:,:,0]))))
    pp1 = ((irpic[1,:,:]*np.exp(-1j*(p0tables[:,:,0]+2*np.pi*2/3))))
    pp2 = ((irpic[2,:,:]*np.exp(-1j*(p0tables[:,:,0]+2*np.pi*4/3))))
    final_phase1 = - np.angle(pp0+pp1+pp2);
    
    
    pp3 = ((irpic[3,:,:]*np.exp(-1j*(p0tables[:,:,1]))))
    pp4 = ((irpic[4,:,:]*np.exp(-1j*(p0tables[:,:,1]+2*np.pi*2/3))))
    pp5 = ((irpic[5,:,:]*np.exp(-1j*(p0tables[:,:,1]+2*np.pi*4/3))))
#    print - np.angle(pp3+pp4+pp5)
#    final_phase2 = - np.angle(pp3+pp4+pp5);

    pp6 = ((irpic[6,:,:]*np.exp(-1j*(p0tables[:,:,2]))))
    pp7 = ((irpic[7,:,:]*np.exp(-1j*(p0tables[:,:,2]+2*np.pi*2/3))))
    pp8 = ((irpic[8,:,:]*np.exp(-1j*(p0tables[:,:,2]+2*np.pi*4/3))))
#    print - np.angle(pp0+pp1+pp2)
#    final_phase3 = - np.angle(pp6+pp7+pp8);

    final_amplitude1 = 2./3. * np.absolute(pp0+pp1+pp2)
    final_amplitude2 = 2./3. * np.absolute(pp2+pp3+pp4)
    final_amplitude3 = 2./3. * np.absolute(pp5+pp6+pp7)


#%%
    fig = plt.figure("Phases and Amplitudes")
    ccp = 'Greens'
    cca = 'Greens_r'

    ax = plt.subplot(3,1,1)
    ax.set_title("RGB image")
    plt.imshow(i)
    
#    for row in range(424):
#        phases[row,:,0] = np.unwrap(phases[row,:,0])
#        phases[row,:,1] = np.unwrap(phases[row,:,1])
#        phases[row,:,2] = np.unwrap(phases[row,:,2])
#        
#    for col in range(512):
#        phases[:,col,0] = np.unwrap(phases[:,col,0])
#        phases[:,col,1] = np.unwrap(phases[:,col,1])
#        phases[:,col,2] = np.unwrap(phases[:,col,2])
    
    ax = plt.subplot(3,4,5)
    ax.set_title("Frequency 1 phase-shift")
    plt.imshow((phases[:,:,0]), cmap=ccp)

    ax = plt.subplot(3,4,6)
    ax.set_title("Frequency 2 phase-shift")
    plt.imshow((phases[:,:,1]), cmap=ccp)

    ax = plt.subplot(3,4,7)
    ax.set_title("Frequency 3 phase-shift")
    plt.imshow((phases[:,:,2]), cmap=ccp)
    
    ax = plt.subplot(3,4,8)
    ax.set_title("Composite phases")
    plt.imshow((phases))

    ax = plt.subplot(3,4,9)
    ax.set_title("Frequency 1 Amplitude")
    plt.imshow(np.minimum(amps[:,:,0],np.median(amps[:,:,0])), cmap=cca)

    ax = plt.subplot(3,4,10)
    ax.set_title("Frequency 2 Amplitude")
    plt.imshow(np.minimum(amps[:,:,1],np.median(amps[:,:,1])), cmap=cca)

    ax = plt.subplot(3,4,11)
    ax.set_title("Frequency 3 Amplitude")
    plt.imshow(np.minimum(amps[:,:,2],np.median(amps[:,:,2])), cmap=cca)
    
    ax = plt.subplot(3,4,12)
    ax.set_title("Composite amplitudes")
    plt.imshow(clrs.rgb_to_hsv(np.minimum(amps,np.median(amps))))

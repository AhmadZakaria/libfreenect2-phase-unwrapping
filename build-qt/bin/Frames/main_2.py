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
    ###ATTEMPT 1 TO CALCULATE WRAPPING COEFICIENTS
    '''
    depth = np.ndarray((424,512),dtype=np.float)
    residuals = np.ndarray((424,512,3),dtype=np.float)
    sum_residuals = np.ndarray((424,512),dtype=np.float)
    
    n_0=1.0
    n_1=1.0
    n_2=1.0
    
    minimum = np.zeros((424,512),dtype=np.float)
    minimum.fill(999999)
    
    n_0chosen = np.zeros((424,512),dtype=np.float)
    
    for y in range (0,424):
        print "iter ",y
        for x in range (0,512):
            for n_0 in np.arange (-5,2,1.0):
                for n_1 in np.arange (-4,4, 1.0):
                    for n_2 in np.arange (-10,11, 1.0):
                        residuals[y,x,0]=3*n_0  -15*n_1 - (15*phases[y,x,1]/(2*np.pi) - 3*phases[y,x,0]/(2*np.pi))
                        residuals[y,x,1]=3*n_0  -2*n_2  - (2*phases[y,x,2]/(2*np.pi)  - 3*phases[y,x,0]/(2*np.pi))
                        residuals[y,x,2]=15*n_1 -2*n_2  - (2*phases[y,x,2]/(2*np.pi)  - 15*phases[y,x,1]/(2*np.pi))
                    
                        sum_residuals[y,x]= np.power(residuals[y,x,0],2) + np.power(residuals[y,x,1],2) + np.power(residuals[y,x,2],2)
                        
                        if sum_residuals[y,x] < minimum[y,x]:
                            #print "changing ",n_0,n_1,n_2
                            minimum[y,x]=sum_residuals[y,x]
                            depth [y,x]=csts.c *(phases[y,x,0] + 2*np.pi *n_0)/(4*np.pi*16)
                            n_0chosen[y,x]=n_0
    
    
    
    
    fig_depth = plt.figure()
#    ax2 = fig_depth.add_subplot(261)
#    ax2.set_title("residuals and sum")
#    #ax2.imshow(np.minimum(ir_filtered,np.median(ir_filtered)), cmap='Greens_r')
#    ax2.imshow(residuals[:,:,0])
#    ax2 = fig_depth.add_subplot(262)
#    ax2.imshow(residuals[:,:,1])
#    ax2 = fig_depth.add_subplot(263)
#    ax2.imshow(residuals[:,:,2])
    
    ax2 = fig_depth.add_subplot(141)
    ax2.imshow(sum_residuals,cmap='Greys')
    ax2 = fig_depth.add_subplot(142)
    ax2.imshow(minimum,cmap='Greys')
    ax2 = fig_depth.add_subplot(143)
    ax2.imshow(depth,cmap='Greys')
    ax2 = fig_depth.add_subplot(144)
    ax2.imshow(n_0chosen,cmap='Greys')
    '''
    
    #%%
    ###ATTEMPT 2 TO GET DEPTH


    #%%
    ###ATTEMPT 1 to implement bilateral filter on amplitude (page 58 of the good pdf)
    amplitude_filtered = np.ndarray((424,512,3),dtype=np.float)
    ir_filtered = np.ndarray((424,512),dtype=np.float)
    
    
    amplitude_filtered[:,:,0]= np.power (-irpic[0,:,:] * np.sin (p0tables[:,:,0]) - irpic[1,:,:] * np.sin (p0tables[:,:,0] + 2*np.pi/3) - irpic[2,:,:] * np.sin (p0tables[:,:,0] + 4*np.pi/3),2) + np.power (irpic[0,:,:] * np.cos (p0tables[:,:,0]) + irpic[1,:,:] * np.cos (p0tables[:,:,0] + 2*np.pi/3) + irpic[2,:,:] * np.cos (p0tables[:,:,0] + 4*np.pi/3),2) 
    amplitude_filtered[:,:,1]= np.power (-irpic[3,:,:] * np.sin (p0tables[:,:,0]) - irpic[4,:,:] * np.sin (p0tables[:,:,0] + 2*np.pi/3) - irpic[5,:,:] * np.sin (p0tables[:,:,0] + 4*np.pi/3),2) + np.power (irpic[3,:,:] * np.cos (p0tables[:,:,0]) + irpic[4,:,:] * np.cos (p0tables[:,:,0] + 2*np.pi/3) + irpic[5,:,:] * np.cos (p0tables[:,:,0] + 4*np.pi/3),2) 
    amplitude_filtered[:,:,2]= np.power (-irpic[6,:,:] * np.sin (p0tables[:,:,0]) - irpic[7,:,:] * np.sin (p0tables[:,:,0] + 2*np.pi/3) - irpic[8,:,:] * np.sin (p0tables[:,:,0] + 4*np.pi/3),2) + np.power (irpic[6,:,:] * np.cos (p0tables[:,:,0]) + irpic[7,:,:] * np.cos (p0tables[:,:,0] + 2*np.pi/3) + irpic[8,:,:] * np.cos (p0tables[:,:,0] + 4*np.pi/3),2)
    
    ir_filtered=(amplitude_filtered[:,:,0]+amplitude_filtered[:,:,1]+amplitude_filtered[:,:,2])/3
    
    #ampli_filt=plt.subplot(5,2,2)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_title("Amplitude composed and filtered")
    ax1.imshow(np.minimum(ir_filtered,np.median(ir_filtered)), cmap='Greens_r')

#%%
    fig = plt.figure("Phases and Amplitudes")
    ccp = 'Greens'
    cca = 'Greens_r'
    
    plt.subplots_adjust(wspace=0, hspace=0)  #Remove spaces between subplots

    ax = plt.subplot(3,1,1)
    ax.set_title("RGB image")
    ax.set_axis_off() #Remove axis for better visualization
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
    ax.set_axis_off() #Remove axis for better visualization
    #ax.set_title("Frequency 1 phase-shift")
    plt.imshow((phases[:,:,0]), cmap=ccp)

    ax = plt.subplot(3,4,6)
    ax.set_axis_off()
    #ax.set_title("Frequency 2 phase-shift")
    plt.imshow((phases[:,:,1]), cmap=ccp)

    ax = plt.subplot(3,4,7)
    ax.set_axis_off()
    #ax.set_title("Frequency 3 phase-shift")
    plt.imshow((phases[:,:,2]), cmap=ccp)
    
    ax = plt.subplot(3,4,8)
    ax.set_axis_off()
    #ax.set_title("Composite phases")
    plt.imshow((phases))

    ax = plt.subplot(3,4,9)
    ax.set_axis_off()
    #ax.set_title("Frequency 1 Amplitude")
    plt.imshow(np.minimum(amps[:,:,0],np.median(amps[:,:,0])), cmap=cca)

    ax = plt.subplot(3,4,10)
    ax.set_axis_off()
    #ax.set_title("Frequency 2 Amplitude")
    plt.imshow(np.minimum(amps[:,:,1],np.median(amps[:,:,1])), cmap=cca)

    ax = plt.subplot(3,4,11)
    ax.set_axis_off()
    #ax.set_title("Frequency 3 Amplitude")
    plt.imshow(np.minimum(amps[:,:,2],np.median(amps[:,:,2])), cmap=cca)
    
    ax = plt.subplot(3,4,12)
    ax.set_axis_off()
    #ax.set_title("Composite amplitudes")
    plt.imshow(clrs.rgb_to_hsv(np.minimum(amps,np.median(amps))))
    
    


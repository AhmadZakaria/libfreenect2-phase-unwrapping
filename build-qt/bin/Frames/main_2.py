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
        p0tables = p0tables.reshape((424,512,3))
        print p0tables.shape
#        plt.imshow(p0tables[:,:,2])



#    with open("/home/ahmad/Documents/Masters/Computational-Photography/project/libfreenect2/build-qt/bin/Frames/BKvbya", "rb") as f2:
    with open("decodedIR", "rb") as f2:


       plt.figure()
       irpic = np.fromfile(f2, dtype=np.int32)
       print (irpic.shape)
       irpic = abs((irpic.reshape((9,424,512))))
       print (irpic.shape)

       plt.subplot(4,3,2)

       i = imread("dumprgb12.jpeg")


       plt.imshow(i)
       counter = 4
       for row in range(2,5):
           for col in range(2,5):
               plt.subplot(4, 3, counter)
               plt.imshow(np.flipud(np.minimum(irpic[counter-4][:][:],np.mean(irpic[counter-4][:][:]))))
               counter = counter +1

#    phases = np.ndarray((424,512,3),dtype=np.float)
#    for x in range (424):
#        for y in range (512):
#            phases[x,y,0] = atan2((-irpic[0,x,y] * sin(p0tables[x,y,0])) - irpic[1,x,y] * sin(p0tables[x,y,0] + 2*np.pi /3 ) -irpic[2,x,y] * sin(p0tables[x,y,0] + 4*np.pi/3), \
#            (-irpic[0,x,y] * cos(p0tables[x,y,0])) - irpic[1,x,y] * cos(p0tables[x,y,0] + 2*np.pi /3 ) -irpic[2,x,y] * cos(p0tables[x,y,0] + 4*np.pi/3) )
#
#            phases[x,y,1] = atan2((-irpic[3,x,y] * sin(p0tables[x,y,1])) - irpic[4,x,y] * sin(p0tables[x,y,1] + 2*np.pi /3 ) -irpic[5,x,y] * sin(p0tables[x,y,1] + 4*np.pi/3), \
#            (-irpic[3,x,y] * cos(p0tables[x,y,1])) - irpic[4,x,y] * cos(p0tables[x,y,1] + 2*np.pi /3 ) -irpic[5,x,y] * cos(p0tables[x,y,1] + 4*np.pi/3) )
#
#            phases[x,y,2] = atan2((-irpic[6,x,y] * sin(p0tables[x,y,2])) - irpic[7,x,y] * sin(p0tables[x,y,2] + 2*np.pi /3 ) -irpic[8,x,y] * sin(p0tables[x,y,2] + 4*np.pi/3), \
#            (-irpic[6,x,y] * cos(p0tables[x,y,2])) - irpic[7,x,y] * cos(p0tables[x,y,2] + 2*np.pi /3 ) -irpic[8,x,y] * cos(p0tables[x,y,2] + 4*np.pi/3) )
#
#    plt.figure()
#    plt.subplot(2,3,2)
#    plt.imshow(i)
#    plt.subplot(2, 3, 4)
#    plt.imshow(np.flipud(phases[:,:,0]))
#    plt.subplot(2, 3, 5)
#    plt.imshow(np.flipud(phases[:,:,1]))
#    plt.subplot(2, 3, 6)
#    plt.imshow(np.flipud(phases[:,:,2]))
#
    pp0 = ((irpic[0,:,:]*np.exp(-1j*(p0tables[:,:,0]))))
    pp1 = ((irpic[1,:,:]*np.exp(-1j*(p0tables[:,:,0]+2*np.pi*2/3))))
    pp2 = ((irpic[2,:,:]*np.exp(-1j*(p0tables[:,:,0]+2*np.pi*4/3))))
#    print - np.angle(pp0+pp1+pp2)
    final_phase1 = - np.angle(pp0+pp1+pp2);
    
    
    
    #%%    
    phases = np.ndarray((424,512,3),dtype=np.float)
    
    first_term = irpic[0,:,:] * np.sin (0) + irpic[1,:,:] * np.sin (2*np.pi/3) + irpic[2,:,:] * np.sin(4*np.pi/3) 
    second_term = irpic[0,:,:] * np.cos(0) + irpic[1,:,:] * np.cos(2*np.pi/3) +  irpic[2,:,:] * np.cos(4*np.pi/3) 
    
    phases[:,:,0]=np.arctan2(-first_term,second_term)
    
    first_term = irpic[3,:,:] * sin (0) + irpic[4,:,:] * sin (2*np.pi/3) + irpic[5,:,:] * sin(4*np.pi/3) 
    second_term = irpic[3,:,:] * cos(0) + irpic[4,:,:] * cos(2*np.pi/3) +  irpic[5,:,:] * cos(4*np.pi/3) 
    
    phases[:,:,1]=np.arctan2(-(first_term),(second_term))
    
    first_term = irpic[6,:,:] * sin (0) + irpic[7,:,:] * sin (2*np.pi/3) + irpic[8,:,:] * sin(4*np.pi/3) 
    second_term = irpic[6,:,:] * cos(0) + irpic[7,:,:] * cos(2*np.pi/3) +  irpic[8,:,:] * cos(4*np.pi/3) 
    
    phases[:,:,2]=np.arctan2(-(first_term),(second_term))
    
    #%%
#    #NEW WAY OF DOING IT ACCORING TO PATENT
#    phases = np.ndarray((424,512,3),dtype=np.float)
#    for x in range (424):
#        for y in range (512):
#            
#            
#            #first phase
#            first_term=0.0
#            for index in range (0,3):
#                first_term=first_term + irpic[index,x,y] * sin (2*np.pi /3 *index)
#            first_term= -first_term
#            
#            second_term=0.0
#            for index in range (0,3):
#                second_term=second_term + irpic[index,x,y] * cos (2*np.pi /3 *index)
#            
#            
#            phase_value=atan2(first_term,second_term)
#            phases[x,y,0]=phase_value
#            
#            
#            
#            #second phase
#            first_term=0.0
#            for index in range (0,3):
#                first_term=first_term + irpic[index+3,x,y] * sin (2*np.pi /3 *index)
#            first_term= -first_term
#            
#            second_term=0.0
#            for index in range (0,3):
#                second_term=second_term + irpic[index+3,x,y] * cos (2*np.pi /3 *index)
#            
#            
#            phase_value=atan2(first_term,second_term)
#            phases[x,y,1]=phase_value
#
#
#
#            #third phase
#            first_term=0.0
#            for index in range (0,3):
#                first_term=first_term + irpic[index+6,x,y] * sin (2*np.pi /3 *index)
#            first_term= -first_term
#            
#            second_term=0.0
#            for index in range (0,3):
#                second_term=second_term + irpic[index+6,x,y] * cos (2*np.pi /3 *index)
#            
#            
#            phase_value=atan2(first_term,second_term)
#            phases[x,y,2]=phase_value            
#            
#                
#            #phases[x,y,0]            
#            
##            phases[x,y,0] = atan2((-irpic[0,x,y] * sin(p0tables[x,y,0])) - irpic[1,x,y] * sin(p0tables[x,y,0] + 2*np.pi /3 ) -irpic[2,x,y] * sin(p0tables[x,y,0] + 4*np.pi/3), \
##            (-irpic[0,x,y] * cos(p0tables[x,y,0])) - irpic[1,x,y] * cos(p0tables[x,y,0] + 2*np.pi /3 ) -irpic[2,x,y] * cos(p0tables[x,y,0] + 4*np.pi/3) )
#
##            phases[x,y,1] = atan2((-irpic[3,x,y] * sin(p0tables[x,y,1])) - irpic[4,x,y] * sin(p0tables[x,y,1] + 2*np.pi /3 ) -irpic[5,x,y] * sin(p0tables[x,y,1] + 4*np.pi/3), \
##            (-irpic[3,x,y] * cos(p0tables[x,y,1])) - irpic[4,x,y] * cos(p0tables[x,y,1] + 2*np.pi /3 ) -irpic[5,x,y] * cos(p0tables[x,y,1] + 4*np.pi/3) )
##
##            phases[x,y,2] = atan2((-irpic[6,x,y] * sin(p0tables[x,y,2])) - irpic[7,x,y] * sin(p0tables[x,y,2] + 2*np.pi /3 ) -irpic[8,x,y] * sin(p0tables[x,y,2] + 4*np.pi/3), \
##            (-irpic[6,x,y] * cos(p0tables[x,y,2])) - irpic[7,x,y] * cos(p0tables[x,y,2] + 2*np.pi /3 ) -irpic[8,x,y] * cos(p0tables[x,y,2] + 4*np.pi/3) )

#%%
#    cmaps = ['viridis', 'inferno', 'plasma', 'magma', 'Blues', 'BuGn', 'BuPu',
#    'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
#    'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
#    'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'bone', 'cool',
#    'copper', 'gist_heat', 'gray', 'hot',
#    'pink', 'spring', 'summer', 'winter', 'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
#    'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
#    'seismic', 'Accent', 'Dark2', 'Paired', 'Pastel1',
#    'Pastel2', 'Set1', 'Set2', 'Set3', 'gist_earth', 'terrain', 'ocean', 'gist_stern',
#    'brg', 'CMRmap', 'cubehelix',
#    'gnuplot', 'gnuplot2', 'gist_ncar',
#    'nipy_spectral', 'jet', 'rainbow',
#    'gist_rainbow', 'hsv', 'flag', 'prism']
#    
#    for cc in cmaps:
    
    cc = 'Greys'
    plt.figure()
    plt.subplot(2,3,2)
    plt.imshow(i)
    plt.subplot(2, 3, 4)
    plt.imshow(np.flipud(phases[:,:,0]), cmap=cc)
    plt.subplot(2, 3, 5)
    plt.imshow(np.flipud(phases[:,:,1]), cmap=cc)
    plt.subplot(2, 3, 6)
    plt.imshow(np.flipud(phases[:,:,2]), cmap=cc)
            
    
#    plt.figure();

    pp3 = ((irpic[3,:,:]*np.exp(-1j*(p0tables[:,:,1]))))
    pp4 = ((irpic[4,:,:]*np.exp(-1j*(p0tables[:,:,1]+2*np.pi*2/3))))
    pp5 = ((irpic[5,:,:]*np.exp(-1j*(p0tables[:,:,1]+2*np.pi*4/3))))
#    print - np.angle(pp3+pp4+pp5)
    final_phase2 = - np.angle(pp3+pp4+pp5);
#    plt.figure();
#    plt.imshow(np.flipud(final_phase2))

    pp6 = ((irpic[6,:,:]*np.exp(-1j*(p0tables[:,:,2]))))
    pp7 = ((irpic[7,:,:]*np.exp(-1j*(p0tables[:,:,2]+2*np.pi*2/3))))
    pp8 = ((irpic[8,:,:]*np.exp(-1j*(p0tables[:,:,2]+2*np.pi*4/3))))
#    print - np.angle(pp0+pp1+pp2)
    final_phase3 = - np.angle(pp6+pp7+pp8);
#    plt.figure();
#    plt.imshow(np.flipud(final_phase3))


#    ind = final_phase3 < 0
#    final_phase3[ind] = final_phase3[ind] + 2. * np.pi

    final_amplitude1 = 2./3. * np.absolute(pp0+pp1+pp2)
    final_amplitude2 = 2./3. * np.absolute(pp2+pp3+pp4)
    final_amplitude3 = 2./3. * np.absolute(pp5+pp6+pp7)

    plt.figure()

    ax = plt.subplot(2,3,1)
    ax.set_title("Frequency 1 phase-shift")
    plt.imshow(np.flipud(final_phase1))

    ax = plt.subplot(2,3,2)
    ax.set_title("Frequency 2 phase-shift")
    plt.imshow(np.flipud(final_phase2))

    ax = plt.subplot(2,3,3)
    ax.set_title("Frequency 3 phase-shift")
    plt.imshow(np.flipud(final_phase3))

    ax = plt.subplot(2,3,4)
    ax.set_title("Frequency 1 Amplitude")
    plt.imshow(np.flipud(np.minimum(final_amplitude1,np.median(final_amplitude1))))

    ax = plt.subplot(2,3,5)
    ax.set_title("Frequency 2 Amplitude")
    plt.imshow(np.flipud(np.minimum(final_amplitude2,np.median(final_amplitude2))))

    ax = plt.subplot(2,3,6)
    ax.set_title("Frequency 3 Amplitude")
    plt.imshow(np.flipud(np.minimum(final_amplitude3,np.median(final_amplitude2))))


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
from math import atan2, sin, cos, floor ,fabs ,log ,exp, sqrt

from skimage.restoration import unwrap_phase
from skimage import exposure
from skimage import filter

ccp = 'Greys_r'
cca = 'Greens_r'


if __name__ == "__main__" :
    print "started"

#    f = open("/home/ahmad/Documents/Masters/Computational-Photography/project/libfreenect2/build-qt/bin/Frames/tab11to16.dat", "r")
#    a = np.fromfile(f, dtype=np.uint16)
#    print a

    with open("p0tables.dat", "rb") as p0tf:
        p0tables = np.fromfile(p0tf, dtype=np.uint16)
        p0tables = np.flipud(p0tables.reshape((424,512,3)))
        print p0tables.shape

    with open("20160616-120945411.ir", "rb") as f2:
       plt.figure("Raw data")
       irpic = np.fromfile(f2, dtype=np.int32)
       print (irpic.shape)
       irpic = (abs(((irpic.reshape((9,424,512))))))
       print (irpic.shape)

       plt.subplot(4,1,1)

       i = imread("20160616-120945332.rgb")


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
    ### ACCORDING TO PATENT, Phases and amplitudes
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
    
    
    #calculate the corrected amplitude according to https://hal.inria.fr/hal-00725654/PDF/TOF.pdf
    distance   = np.ndarray((424,512,3),dtype=np.float)
    
    distance[:,:,0]=csts.c*phases[:,:,0]/(4*np.pi*16)
    distance[:,:,1]=csts.c*phases[:,:,1]/(4*np.pi*80)
    distance[:,:,2]=csts.c*phases[:,:,2]/(4*np.pi*120)
#    
#    amps[:,:,0]=amps[:,:,0]*np.power(distance[:,:,0],2)
#    amps[:,:,1]=amps[:,:,1]*np.power(distance[:,:,1],2)
#    amps[:,:,2]=amps[:,:,2]*np.power(distance[:,:,2],2)
    
#    amps[:,:,0]=np.minimum(amps[:,:,0],np.median(amps[:,:,0]))
#    amps[:,:,1]=np.minimum(amps[:,:,1],np.median(amps[:,:,1]))
#    amps[:,:,2]=np.minimum(amps[:,:,2],np.median(amps[:,:,2]))
#    
#    amps[:,:,0] = exposure.equalize_hist(amps[:,:,0])    
#    amps[:,:,1] = exposure.equalize_hist(amps[:,:,1])    
#    amps[:,:,2] = exposure.equalize_hist(amps[:,:,2]) 
    
#    amps[:,:,0]=distance[:,:,0]
#    amps[:,:,1]=distance[:,:,1]
#    amps[:,:,2]=distance[:,:,2]
    
    

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
    ###ATTEMPT 1 to implement bilateral filter on amplitude (page 58 of the good pdf)
    amplitude_filtered = np.ndarray((424,512,3),dtype=np.float)
    ir_filtered = np.ndarray((424,512),dtype=np.float)
    
    
    amplitude_filtered[:,:,0]= np.power (-irpic[0,:,:] * np.sin (p0tables[:,:,0]) - irpic[1,:,:] * np.sin (p0tables[:,:,0] + 2*np.pi/3) - irpic[2,:,:] * np.sin (p0tables[:,:,0] + 4*np.pi/3),2) + np.power (irpic[0,:,:] * np.cos (p0tables[:,:,0]) + irpic[1,:,:] * np.cos (p0tables[:,:,0] + 2*np.pi/3) + irpic[2,:,:] * np.cos (p0tables[:,:,0] + 4*np.pi/3),2) 
    amplitude_filtered[:,:,1]= np.power (-irpic[3,:,:] * np.sin (p0tables[:,:,1]) - irpic[4,:,:] * np.sin (p0tables[:,:,1] + 2*np.pi/3) - irpic[5,:,:] * np.sin (p0tables[:,:,1] + 4*np.pi/3),2) + np.power (irpic[3,:,:] * np.cos (p0tables[:,:,1]) + irpic[4,:,:] * np.cos (p0tables[:,:,1] + 2*np.pi/3) + irpic[5,:,:] * np.cos (p0tables[:,:,1] + 4*np.pi/3),2) 
    amplitude_filtered[:,:,2]= np.power (-irpic[6,:,:] * np.sin (p0tables[:,:,2]) - irpic[7,:,:] * np.sin (p0tables[:,:,2] + 2*np.pi/3) - irpic[8,:,:] * np.sin (p0tables[:,:,2] + 4*np.pi/3),2) + np.power (irpic[6,:,:] * np.cos (p0tables[:,:,2]) + irpic[7,:,:] * np.cos (p0tables[:,:,2] + 2*np.pi/3) + irpic[8,:,:] * np.cos (p0tables[:,:,2] + 4*np.pi/3),2)
    
    #ir_filtered=(amplitude_filtered[:,:,0]+amplitude_filtered[:,:,1]+amplitude_filtered[:,:,2])/3
    ir_filtered=(amps[:,:,0]+amps[:,:,1]+amps[:,:,2])/3
    
    #ampli_filt=plt.subplot(5,2,2)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_title("Amplitude composed and filtered")
    
    
    #ir_filtered = exposure.equalize_hist(ir_filtered)   
    #ir_filtered=np.minimum(ir_filtered,np.mean(ir_filtered))
    ax1.imshow(ir_filtered, cmap='Greens_r')
    
    
    #print ir_filtered
    
    #%%
    ###ATTEMPT 2 TO GET DEPTH
    '''
    depth = np.ndarray((424,512),dtype=np.float)
    depth.fill(0.0)
    
    ab_confidence_slope= -0.5330578
    ab_confidence_offset = 0.7694894
    max_dealias_confidence = 0.6108653
    min_dealias_confidence = 0.3490659
    phase_offset = 0.0
    unambigious_dist = 2083.333
    individual_ab_threshold  = 3.0
    ab_threshold = 10.0;
    
    
    #Make the xz tables
    TABLE_SIZE= 512*424
    scaling_factor = 8192
    unambigious_dist = 6250.0/3
    divergence = 0
    x_table = np.ndarray((TABLE_SIZE),dtype=np.float)
    z_table = np.ndarray((TABLE_SIZE),dtype=np.float)

    #found these values here https://threeconstants.wordpress.com/2014/11/09/kinect-v2-depth-camera-calibration/
    fx=391.096
    fy=463.098    
    
    cx=243.892
    cy= 208.922
    
    for iter in range(0,TABLE_SIZE):
        xi = iter % 512;
        yi = iter / 512;
        xd = (xi + 0.5 - cx)/fx;
        yd = (yi + 0.5 - cy)/fy;
        #double xu, yu;
        #divergence += !undistort(xd, yd, xu, yu);
        x_table[iter] = scaling_factor*xd;
        z_table[iter] = unambigious_dist/sqrt(xd*xd + yd*yd + 1);
    

    
    for y in range (0,424):
        ##print "iter ",y
        for x in range (0,512):
            
            ir_min = min(min(amplitude_filtered[y,x,0], amplitude_filtered[y,x,1]), amplitude_filtered[y,x,2])
            ir_sum = amplitude_filtered[y,x,0] + amplitude_filtered[y,x,1] + amplitude_filtered[y,x,2];
        
            if ir_min < individual_ab_threshold or ir_sum < ab_threshold:
                phase = 0;            
            
            else:
                t0 = phases[y,x,0] / (2.0 * np.pi) * 3.0;
                t1 = phases[y,x,1] / (2.0 * np.pi) * 15.0;
                t2 = phases[y,x,2] / (2.0 * np.pi) * 2.0;
                
                t5 = (floor((t1 - t0) * 0.333333 + 0.5) * 3.0 + t0)
                t3 = (-t2 + t5)
                t4 = t3 * 2.0
                
                c1 = (t4 >= -t4) #True if t4 is positive
                
                
                f1=0.0
                if c1:
                    f1=2.0
                    f2=0.5
                else:
                    f1=-2.0
                    f2=-0.5
                t3=t3*f2
                t3 = (t3 - floor(t3)) * f1;
                
                c2 = ((0.5 < abs(t3)) and (abs(t3) < 1.5))
                
                t6=0.0
                if c1:
                    t6=t5 + 15.0
                    t7=t1 + 15.0
                else:
                    t6=t5
                    t7=t1
                
                t8 = (floor((-t2 + t6) * 0.5 + 0.5) * 2.0 + t2) * 0.5
                
                t6 = t6* 0.333333
                t7 = t7* 0.066667
                
                t9 = (t8 + t6 + t7)
                t10 = t9 * 0.333333 # some avg
                
                t6 =t6* 2.0 * np.pi
                t7 =t7* 2.0 * np.pi
                t8 =t8* 2.0 * np.pi
                
                # some cross product
                t8_new = t7 * 0.826977 - t8 * 0.110264
                t6_new = t8 * 0.551318 - t6 * 0.826977
                t7_new = t6 * 0.110264 - t7 * 0.551318
                
                t8 = t8_new
                t6 = t6_new
                t7 = t7_new
                
                norm = t8 * t8 + t6 * t6 + t7 * t7
                mask=0.0
                if t9 >= 0.0:
                    mask=1.0
                else:
                    mask=0.0
    
                t10 =t10* mask
                
                
                slope_positive = 0 < ab_confidence_slope
                
                ir_min_ = min(min(amplitude_filtered[y,x,0], amplitude_filtered[y,x,1]), amplitude_filtered[y,x,2]);
                ir_max_ = max(max(amplitude_filtered[y,x,0], amplitude_filtered[y,x,1]), amplitude_filtered[y,x,2]);
                
                ir_x=0.0
                if slope_positive:
                    #print "Slope_positive is true and ir_min is",ir_min_
                    ir_x=ir_min_
                else:
                    #print "Slope_positive is false and ir_max_ is",ir_max_
                    ir_x=ir_max_
                
                #If it's 0 continue because the log(0) will fail
                if ir_x==0:
                    continue
                
                #print "ir_x",ir_x
                ir_x = log(ir_x);
                ir_x = (ir_x * ab_confidence_slope * 0.301030 + ab_confidence_offset) * 3.321928;
                ir_x = exp(ir_x);
                ir_x = min(max_dealias_confidence, max(min_dealias_confidence, ir_x));
                ir_x =ir_x* ir_x;
                
                mask=0.0
                if ir_x >= norm:
                    mask2=1.0
                else:
                    mask2=0.0
                
                t11 = t10 * mask2;
                
                mask3=0.0
                if max_dealias_confidence * max_dealias_confidence >= norm:
                    mask3=1.0
                else:
                    mask3=0.0
                    
                t10 = t10* mask3
                phase = t11
                
                
                
                
            #I don't think the tables are correct
            x_table = x_table.reshape(424,512)
            z_table = z_table.reshape(424,512)
            
            zmultiplier = z_table[y, x]
            xmultiplier = x_table[y, x]
            
            if 0 < phase:
                phase=phase + phase_offset
            else:
                phase=phase
                
            depth_linear = zmultiplier * phase;
            max_depth = phase * unambigious_dist * 2;
            
            cond1 =  True and (0 < depth_linear) and (0 < max_depth)
            
            xmultiplier = (xmultiplier * 90) / (max_depth * max_depth * 8192.0)
            
            depth_fit = depth_linear / (-depth_linear * xmultiplier + 1)
            
            if depth_fit < 0:
                depth_fit=0.0
            else:
                depth_fit=depth_fit
              
            depth_final=0.0
            if cond1:
                depth_final=depth_fit
            else:
                depth_final=depth_linear
            
            
            depth[y,x]=t2
            
    fig_depth = plt.figure()
    ax_depth = fig_depth.add_subplot(111)   
    ax_depth.imshow(depth, cmap='Greys_r')        
            

    print "Finished depth attempt 2"
    '''
    
    #%%
    '''
    def undistort(x,y):
        
    x0 = x;
    y0 = y;

    last_x = x;
    last_y = y;
    max_iterations = 100;
    
    for iter in range (0,max_iterations):
      x2 = x*x;
      y2 = y*y;
      x2y2 = x2 + y2;
      x2y22 = x2y2*x2y2;
      x2y23 = x2y2*x2y22;

      #Jacobian matrix
      Ja = k3*x2y23 + (k2+6*k3*x2)*x2y22 + (k1+4*k2*x2)*x2y2 + 2*k1*x2 + 6*p2*x + 2*p1*y + 1;
      Jb = 6*k3*x*y*x2y22 + 4*k2*x*y*x2y2 + 2*k1*x*y + 2*p1*x + 2*p2*y;
      Jc = Jb;
      double Jd = k3*x2y23 + (k2+6*k3*y2)*x2y22 + (k1+4*k2*y2)*x2y2 + 2*k1*y2 + 2*p2*x + 6*p1*y + 1;

      //Inverse Jacobian
      double Jdet = 1/(Ja*Jd - Jb*Jc);
      double a = Jd*Jdet;
      double b = -Jb*Jdet;
      double c = -Jc*Jdet;
      double d = Ja*Jdet;

      double f, g;
      distort(x, y, f, g);
      f -= x0;
      g -= y0;

      x -= a*f + b*g;
      y -= c*f + d*g;
      const double eps = std::numeric_limits<double>::epsilon()*16;
      if (fabs(x - last_x) <= eps && fabs(y - last_y) <= eps)
        break;
      last_x = x;
      last_y = y;
    }
    xu = x;
    yu = y;
    return iter < max_iterations;
    '''
    
    
    #%%    
    ###ATTEMPT 3 TO CALCULATE WRAPPING COEFICIENTS
    '''
    depth = np.ndarray((424,512),dtype=np.float)
    residuals = np.ndarray((424,512,3),dtype=np.float)
    sum_residuals = np.ndarray((424,512),dtype=np.float)
    
    n_0=1.0
    n_1=1.0
    n_2=1.0
    
    t0=3*phases[:,:,0]/(2*np.pi)
    t1=15*phases[:,:,1]/(2*np.pi)
    t2=2*phases[:,:,2]/(2*np.pi)
    
    minimum = np.zeros((424,512),dtype=np.float)
    minimum.fill(999999)
    
    residuals[:,:,0]=3*n_0  -15*n_1 - (15*phases[:,:,1]/(2*np.pi) - 3*phases[:,:,0]/(2*np.pi))
    residuals[:,:,1]=3*n_0  -2*n_2  - (2*phases[:,:,2]/(2*np.pi)  - 3*phases[:,:,0]/(2*np.pi))
    residuals[:,:,2]=15*n_1 -2*n_2  - (2*phases[:,:,2]/(2*np.pi)  - 15*phases[:,:,1]/(2*np.pi))
    
    t1_t0_3=(t1-t0)/3
    t2_t0=t2-t0
    t2_t1=t2-t1
    
    
    
    fig_depth = plt.figure()
    ax2 = fig_depth.add_subplot(131)
    ax2.set_title("residuals and sum")
    #ax2.imshow(np.minimum(ir_filtered,np.median(ir_filtered)), cmap='Greens_r')
    ax2.imshow(t1_t0_3,cmap='Greys')
    ax2 = fig_depth.add_subplot(132)
    ax2.imshow(t1,cmap='Greys')
    ax2 = fig_depth.add_subplot(133)
    ax2.imshow(t2_t1,cmap='Greys')
    '''
    def distort (x,y):
        
        cx=255.642
        cy=203.704
        fx=365.8
        fy=365.8
        k1=0.091391
        k2=-0.271139
        k3=0.0950107
        p1=0.0
        p2=0.0
        
        x2 = x * x;
        y2 = y * y;
        r2 = x2 + y2;
        xy = x * y;
        kr = ((k3 * r2 + k2) * r2 + k1) * r2 + 1.0;
        
        xd = x*kr + p2*(r2 + 2*x2) + 2*p1*xy;
        yd = y*kr + p1*(r2 + 2*y2) + 2*p2*xy;
        
        return xd,yd
    
    def undistort (x,y):
        cx=255.642
        cy=203.704
        fx=365.8
        fy=365.8
        k1=0.091391
        k2=-0.271139
        k3=0.0950107
        p1=0.0
        p2=0.0
            
        
        x0 = x
        y0 = y
    
        last_x = x;
        last_y = y;
        max_iterations = 100;
        iter;
        for inter in range (0, max_iterations):
          x2 = x*x;
          y2 = y*y;
          x2y2 = x2 + y2;
          x2y22 = x2y2*x2y2;
          x2y23 = x2y2*x2y22;
    
          #Jacobian matrix
          Ja = k3*x2y23 + (k2+6*k3*x2)*x2y22 + (k1+4*k2*x2)*x2y2 + 2*k1*x2 + 6*p2*x + 2*p1*y + 1;
          Jb = 6*k3*x*y*x2y22 + 4*k2*x*y*x2y2 + 2*k1*x*y + 2*p1*x + 2*p2*y;
          Jc = Jb;
          Jd = k3*x2y23 + (k2+6*k3*y2)*x2y22 + (k1+4*k2*y2)*x2y2 + 2*k1*y2 + 2*p2*x + 6*p1*y + 1;
    
          #//Inverse Jacobian
          Jdet = 1/(Ja*Jd - Jb*Jc);
          a = Jd*Jdet;
          b = -Jb*Jdet;
          c = -Jc*Jdet;
          d = Ja*Jdet;
    
          f=0.0
          g=0.0
          f,g=distort(x, y);
          f =f - x0;
          g =g - y0;
    
          x =x- a*f + b*g;
          y =y- c*f + d*g;
          
        xu = x;
        yu = y;
        return xu,yu
    #%%    
    ###ATTEMPT 4 TO CALCULATE WRAPPING COEFICIENTS
    
    '''
    ab_confidence_slope= -0.5330578
    ab_confidence_offset = 0.7694894
    max_dealias_confidence = 0.6108653
    min_dealias_confidence = 0.3490659
    phase_offset = 0.0
    unambigious_dist = 2083.333
    individual_ab_threshold  = 3.0
    ab_threshold = 10.0;
    
    depth = np.ndarray((424,512),dtype=np.float)
    residuals = np.ndarray((424,512,3),dtype=np.float)
    sum_residuals = np.ndarray((424,512),dtype=np.float)
    
    n_0=1.0
    n_1=1.0
    n_2=1.0
    
    t0 = np.ndarray((424,512),dtype=np.float)
    t1 = np.ndarray((424,512),dtype=np.float)
    t2 = np.ndarray((424,512),dtype=np.float)
    t3 = np.ndarray((424,512),dtype=np.float)
    t5 = np.ndarray((424,512),dtype=np.float)
    t4 = np.ndarray((424,512),dtype=np.float)
    
    t6 = np.ndarray((424,512),dtype=np.float)
    t7 = np.ndarray((424,512),dtype=np.float)
    t8 = np.ndarray((424,512),dtype=np.float)
    t9 = np.ndarray((424,512),dtype=np.float)
    t10 = np.ndarray((424,512),dtype=np.float)
    t11 = np.ndarray((424,512),dtype=np.float)
    
    t6_new = np.ndarray((424,512),dtype=np.float)
    t7_new = np.ndarray((424,512),dtype=np.float)
    t8_new = np.ndarray((424,512),dtype=np.float)
    
    ir_max_ = np.ndarray((424,512),dtype=np.float)
    ir_x = np.ndarray((424,512),dtype=np.float)
    
    phase = np.ndarray((424,512),dtype=np.float)

    
    cx=255.642
    cy=203.704
    fx=365.8
    fy=365.8
    k1=0.091391
    k2=-0.271139
    k3=0.0950107
    p1=0.0
    p2=0.0
    
    
    
    
    #Make the xz tables
    TABLE_SIZE= 512*424
    scaling_factor = 8192
    unambigious_dist = 6250.0/3
    divergence = 0
    x_table = np.ndarray((TABLE_SIZE),dtype=np.float)
    z_table = np.ndarray((TABLE_SIZE),dtype=np.float)

    print "started with the xz tables"
    
    for iter in range(0,TABLE_SIZE):
        xi = iter % 512;
        yi = iter / 512;
        xd = (xi + 0.5 - cx)/fx;
        yd = (yi + 0.5 - cy)/fy;
        
        #print iter
        xu, yu=undistort(xd, yd);
        x_table[iter] = scaling_factor*xd;
        z_table[iter] = unambigious_dist/sqrt(xd*xd + yd*yd + 1);    
    
    
    x_table = x_table.reshape(424,512)
    z_table = z_table.reshape(424,512)
    
    print "finished with the xz tables"
    
    for y in range (0,424):
        for x in range (0,512):
            ir_min = min(min(amps[y,x,0], amps[y,x,1]), amps[y,x,2])
            ir_sum = amps[y,x,0] + amps[y,x,1] + amps[y,x,2];
        
            if ir_min < individual_ab_threshold or ir_sum < ab_threshold:
                phase[y,x] = 0;            
            
            else:
                
                #I believe the initial guess of t0.t1 and t2 are not correct since the oder is may be 80, 16 and then 120 mhz
#                t0[y,x]=3*phases[y,x,0]/(2*np.pi)
#                t1[y,x]=15*phases[y,x,1]/(2*np.pi)
#                t2[y,x]=2*phases[y,x,2]/(2*np.pi)
                
                t0[y,x]=3*phases[y,x,1]/(2*np.pi)
                t1[y,x]=15*phases[y,x,0]/(2*np.pi)
                t2[y,x]=2*phases[y,x,2]/(2*np.pi)
                
                t5[y,x] = (floor((t1[y,x] - t0[y,x]) * 0.333333 + 0.5) * 3.0 + t0[y,x]);
                t3[y,x] = (-t2[y,x] + t5[y,x]);
                t4[y,x] = t3[y,x] * 2.0;
                
                c1= t4[y,x] >= -t4[y,x]
                
                #print c1
                
                f1=0.0
                if c1:
                    f1=2.0
                    f2=0.5
                else:
                    f1=-2.0
                    f2=-0.5
                    
                t3[y,x] =t3[y,x]* f2
                t3[y,x] = (t3[y,x] - floor(t3[y,x])) * f1
                
                c2 = 0.5 < abs(t3[y,x]) and abs(t3[y,x]) < 1.5;
                
                #print c2
                if c2:
                    t6[y,x]=t5[y,x] + 15.0
                    t7[y,x]=t1[y,x] + 15.0
                else:
                    t6[y,x]=t5[y,x]
                    t7[y,x]=t1[y,x]
                    
                t8[y,x] = (floor((-t2[y,x] + t6[y,x]) * 0.5 + 0.5) * 2.0 + t2[y,x]) * 0.5
                
                t6[y,x] =t6[y,x] * 0.333333
                t7[y,x] =t7[y,x] *0.066667
                
                t9[y,x] = (t8[y,x] + t6[y,x] + t7[y,x]); 
                t10[y,x] = t9[y,x] * 0.333333
                
                t6[y,x] =t6[y,x]* 2.0 * np.pi;
                t7[y,x] =t7[y,x]* 2.0 * np.pi;
                t8[y,x] =t8[y,x]* 2.0 * np.pi;
                
                t8_new[y,x] = t7[y,x] * 0.826977 - t8[y,x] * 0.110264
                t6_new[y,x] = t8[y,x] * 0.551318 - t6[y,x] * 0.826977
                t7_new[y,x] = t6[y,x] * 0.110264 - t7[y,x] * 0.551318
                
                t8[y,x] = t8_new[y,x];
                t6[y,x] = t6_new[y,x];
                t7[y,x] = t7_new[y,x];
                
                norm = t8[y,x] * t8[y,x] + t6[y,x] * t6[y,x] + t7[y,x] * t7[y,x];
                
                mask=0.0
                if t9[y,x]>= 0.0:
                    mask=1.0
        
                #print mask
                t10[y,x] =t10[y,x]* mask;
                
                
                #No need for the slope becuse it is always false
                ir_max_[y,x] = max(max(amps[y,x,0], amps[y,x,1]), amps[y,x,2]);
                ir_x[y,x]=ir_max_[y,x]
                
                
                
                ir_x[y,x] = log(ir_x[y,x]);
                ir_x[y,x] = (ir_x[y,x] * ab_confidence_slope * 0.301030 + ab_confidence_offset) * 3.321928;
                ir_x[y,x] = exp(ir_x[y,x]);
                ir_x[y,x] = min(max_dealias_confidence, max(min_dealias_confidence, ir_x[y,x]));
                ir_x[y,x] = ir_x[y,x]*ir_x[y,x];
                
                mask2=0.0
                if ir_x[y,x]>= norm:
                    mask2=1.0
                    
                
                t11[y,x] = t10[y,x] * mask2
                
                
                mask3=0.0
                if max_dealias_confidence*max_dealias_confidence>= norm:
                    mask3=1.0
                t10[y,x] = t10[y,x]*mask3
                
                phase[y,x]=t11[y,x]
            
            zmultiplier = z_table[y, x]
            xmultiplier = x_table[y, x]
            
            
            
            if 0 < phase[y,x]:
                phase[y,x]=phase[y,x] + phase_offset
                
            depth_linear = zmultiplier * phase[y,x] ;
            max_depth = phase[y,x]  * unambigious_dist * 2;
            
            cond1 =  True and (0 < depth_linear) and (0 < max_depth)
            
            xmultiplier = (xmultiplier * 90) / (max_depth * max_depth * 8192.0)
            
            depth_fit = depth_linear / (-depth_linear * xmultiplier + 1)
            
            if depth_fit < 0:
                depth_fit=0.0
            else:
                depth_fit=depth_fit
              
            depth_final=0.0
            if cond1:
                depth_final=depth_fit
            else:
                depth_final=depth_linear
            
            
            depth[y,x]=depth_final
            
            
                
   
#    
#    residuals[:,:,0]=3*n_0  -15*n_1 - (15*phases[:,:,1]/(2*np.pi) - 3*phases[:,:,0]/(2*np.pi))
#    residuals[:,:,1]=3*n_0  -2*n_2  - (2*phases[:,:,2]/(2*np.pi)  - 3*phases[:,:,0]/(2*np.pi))
#    residuals[:,:,2]=15*n_1 -2*n_2  - (2*phases[:,:,2]/(2*np.pi)  - 15*phases[:,:,1]/(2*np.pi))
#    
#    t1_t0_3=(t1-t0)/3
#    t2_t0=t2-t0
#    t2_t1=t2-t1
#    
#    
    
    fig_depth = plt.figure()
    ax2 = fig_depth.add_subplot(131)
    ax2.set_title("residuals and sum")
    #depth=np.minimum(depth,np.median(depth))
    #ax2.imshow(np.minimum(ir_filtered,np.median(ir_filtered)), cmap='Greens_r')
    ax2.imshow(depth,cmap='Greys')
    ax2 = fig_depth.add_subplot(132)
    ax2.imshow(z_table,cmap='Greys')
    ax2 = fig_depth.add_subplot(133)
    ax2.imshow(ir_x,cmap='Greys')
    '''
    
#%%
    fig = plt.figure("Phases and Amplitudes")
    
    
    plt.subplots_adjust(wspace=0, hspace=0)  #Remove spaces between subplots

    ax = plt.subplot(4,1,1)
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
    
#    phases[:,:,0]=filter.gaussian_filter(phases[:,:,0],3)
#    phases[:,:,1]=filter.gaussian_filter(phases[:,:,1],3)
#    phases[:,:,2]=filter.gaussian_filter(phases[:,:,2],3)
    
#    phases[:,:,0]=filter.denoise_bilateral(abs(phases[:,:,0]), sigma_range=0.05, sigma_spatial=5)
#    phases[:,:,1]=filter.denoise_bilateral(abs(phases[:,:,1]), sigma_range=0.05, sigma_spatial=5)
#    phases[:,:,2]=filter.denoise_bilateral(abs(phases[:,:,2]), sigma_range=0.05, sigma_spatial=5)
    
    
    

#    phases[:,:,0] = exposure.equalize_hist(phases[:,:,0])    
#    phases[:,:,1] = exposure.equalize_hist(phases[:,:,1])
#    phases[:,:,2] = exposure.equalize_hist(phases[:,:,2])
    
    
    
    ax = plt.subplot(4,4,5)
    ax.set_axis_off() #Remove axis for better visualization
    #ax.set_title("Frequency 1 phase-shift")
    plt.imshow((phases[:,:,0]), cmap=ccp)

    ax = plt.subplot(4,4,6)
    ax.set_axis_off()
    #ax.set_title("Frequency 2 phase-shift")
    plt.imshow((phases[:,:,1]), cmap=ccp)

    ax = plt.subplot(4,4,7)
    ax.set_axis_off()
    #ax.set_title("Frequency 3 phase-shift")
    plt.imshow((phases[:,:,2]), cmap=ccp)
    
    ax = plt.subplot(4,4,8)
    ax.set_axis_off()
    #ax.set_title("Composite phases")
    plt.imshow(( (phases[:,:,0]+phases[:,:,1]+phases[:,:,2])/3 ))

    ax = plt.subplot(4,4,9)
    ax.set_axis_off()
    #ax.set_title("Frequency 1 Amplitude")
    plt.imshow(amps[:,:,0], cmap=cca)

    ax = plt.subplot(4,4,10)
    ax.set_axis_off()
    #ax.set_title("Frequency 2 Amplitude")
    plt.imshow(amps[:,:,1], cmap=cca)

    ax = plt.subplot(4,4,11)
    ax.set_axis_off()
    #ax.set_title("Frequency 3 Amplitude")
    plt.imshow(amps[:,:,2], cmap=cca)
    
    ax = plt.subplot(4,4,12)
    ax.set_axis_off()
    #ax.set_title("Composite amplitudes")
    plt.imshow(clrs.rgb_to_hsv(np.minimum(amps,np.median(amps))))
    
    
    
    #Showing the unrwapped ones
    phases_unwrapped = np.ndarray((424,512,3),dtype=np.float)
    phases_unwrapped[:,:,0] = unwrap_phase(phases[:,:,0])
    phases_unwrapped[:,:,1] = unwrap_phase(phases[:,:,1])
    phases_unwrapped[:,:,2] = unwrap_phase(phases[:,:,2])
    
    phases_unwrapped[:,:,0] = exposure.equalize_hist(phases_unwrapped[:,:,0])    
    phases_unwrapped[:,:,1] = exposure.equalize_hist(phases_unwrapped[:,:,1])
    phases_unwrapped[:,:,2] = exposure.equalize_hist(phases_unwrapped[:,:,2])
    
    
    
    
    ax = plt.subplot(4,4,13)
    ax.set_axis_off()
    #ax.set_title("Composite amplitudes")
    plt.imshow(phases_unwrapped[:,:,0], cmap=ccp)
    
    ax = plt.subplot(4,4,14)
    ax.set_axis_off()
    #ax.set_title("Composite amplitudes")
    plt.imshow(phases_unwrapped[:,:,1], cmap=ccp)
    
    ax = plt.subplot(4,4,15)
    ax.set_axis_off()
    #ax.set_title("Composite amplitudes")
    plt.imshow(phases_unwrapped[:,:,2], cmap=ccp)
    
    ax = plt.subplot(4,4,16)
    ax.set_axis_off()
    #ax.set_title("Composite amplitudes")
    plt.imshow(( (phases_unwrapped[:,:,0]+phases_unwrapped[:,:,1]+phases_unwrapped[:,:,2])/3 ))
    
    


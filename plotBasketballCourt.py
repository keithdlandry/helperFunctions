#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:58:55 2017

@author: keith.landry

plots a half court basketball court on a 3d matplotlib axis
"""
import numpy as np

def plot_basketball_court(ax, min_h, full_court = False):

    if full_court:
        maxX = 94
    else:
        maxX= 47
    ax.set_xlim3d([0,maxX]) # 94 or 47
    ax.set_ylim3d([0,50])
    ax.set_zlim3d([min_h,17])
    
    hoop = np.array([5.25, 25, 10])
    
    cornerX = np.arange(0,15,1)
    leftCornerY = [3]*len(cornerX)
    rightCornerY = [50 - 3]*len(cornerX)
    
    curveX = np.arange(14, 23.75+5.25+.25, .25) # 23.75 is distance of straight away three # center of hoop is 5.25 from baseline
    curveY = np.sqrt(23.75**2 - (curveX-hoop[0])**2) + hoop[1]
    curveY2 = hoop[1] - np.sqrt(23.75**2 - (curveX-hoop[0])**2)

    ax.plot(curveX, curveY, min_h, c = 'black')
    ax.plot(curveX, curveY2, min_h, c = 'black')
    ax.plot(cornerX, leftCornerY, min_h, c = 'black')
    ax.plot(cornerX, rightCornerY, min_h, c = 'black')
    
    laneX = np.arange(0,20,1)
    laneY = [25-6]*len(laneX)
    laneY2 = [25+6]*len(laneX)
    ax.plot(laneX, laneY, min_h, c = 'black')
    ax.plot(laneX, laneY2, min_h, c = 'black')
    
    freeThrowY = np.arange(25-6,25+6+1,1)
    freeThrowX = [19]*len(freeThrowY)
    ax.plot(freeThrowX, freeThrowY, min_h, c = 'black')
    
    postZ = np.arange(min_h, 11, 1)
    postX = [0]*len(postZ) 
    postY = [25]*len(postZ)
    ax.plot(postX, postY, postZ, c = 'black')
    
    barX = np.arange(0,5,.5) # back of rim at 4.5
    barY = [25]*len(barX)
    ax.plot(barX, barY, 10, c = 'black')
    
    xrim = np.arange(4.5, 6.05,.05)
    yrim = np.sqrt(.75**2 - (xrim-hoop[0])**2) + hoop[1]
    yrim2 = hoop[1] - np.sqrt(.75**2 - (xrim-hoop[0])**2)
    ax.plot(xrim, yrim, 10, c = 'orange', zorder = 1)
    ax.plot(xrim, yrim2, 10, c = 'orange', zorder = 4)
    
    baselineY = np.arange(0,51,1)
    baselineX = np.array([0]*len(baselineY))
    ax.plot(baselineX, baselineY, min_h, c = 'black')

    sidelineX = np.arange(0,48,1)
    sidelineY = np.array([0]*len(sidelineX))
    sidelineY2 = np.array([50]*len(sidelineX))
    ax.plot(sidelineX, sidelineY, min_h, c = 'black')
    ax.plot(sidelineX, sidelineY2, min_h, c = 'black')
    
    halfX = np.array([47]*len(baselineY))
    ax.plot(halfX, baselineY, min_h, c = 'black')

    # create second half by mirroring first half 
    if full_court:
        cornerX2 = -1*cornerX + 94
        curveX2 = -1*curveX + 94
        ax.plot(curveX2, curveY, min_h, c = 'black')
        ax.plot(curveX2, curveY2, min_h, c = 'black')
        ax.plot(cornerX2, leftCornerY, min_h, c = 'black')
        ax.plot(cornerX2, rightCornerY, min_h, c = 'black')
                
        laneX2 = -1*laneX + 94
        ax.plot(laneX2, laneY, min_h, c = 'black')
        ax.plot(laneX2, laneY2, min_h, c = 'black')

        freeThrowX2 = -1*np.array(freeThrowX) + 94
        ax.plot(freeThrowX2, freeThrowY, min_h, c = 'black')

        barX2 = -1*barX + 94
        ax.plot(barX2, barY, 10, c = 'black')

        postX2 = -1*np.array(postX) + 94   
        ax.plot(postX2, postY, postZ, c = 'black')

        xrim2 = -1*xrim + 94
        ax.plot(xrim2, yrim, 10, c = 'orange', zorder = 1)
        ax.plot(xrim2, yrim2, 10, c = 'orange', zorder = 4)
        
        baselineX2 = -1*baselineX + 94
        ax.plot(baselineX2, baselineY, min_h, c = 'black')

        sidelineX2 = -1*sidelineX + 94
        ax.plot(sidelineX2, sidelineY, min_h, c = 'black')
        ax.plot(sidelineX2, sidelineY2, min_h, c = 'black')

# example of use
        
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from plotBasketballCourt import plot_basketball_court
#  
#fig = plt.figure()
#fig.set_size_inches(10, 15)
#ax = plt.gca(projection='3d')
#ax.set_xlim3d([0,94]) # 47 for half court, 94 for full court
#ax.set_ylim3d([0,50])
#
#ax.set_zlim3d([0,17])
#
#plot_basketball_court(ax, min_h=0, full_court=True)   
#plt.show()
 

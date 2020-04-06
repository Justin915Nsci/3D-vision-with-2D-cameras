# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:19:52 2020

@author: justi
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    imgL = cv2.imread('pics/im0.jpg',1)
    imgR = cv2.imread('pics/im1.jpg',1)
    
   
    imgL_gry = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gry = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgL_gry',imgL_gry)
    #cv2.waitKey(0)
    windowSize = 4
    leftFeatures = findInterestRegions(imgL_gry,windowSize)
    #print(len(leftFeatures))
    rightFeatures =  findInterestRegions(imgR_gry,windowSize)
    #print(len(rightFeatures))
    disparityMap = getDisparityMap(rightFeatures,leftFeatures,imgL_gry,imgR_gry,windowSize)
    for i in disparityMap:
        print(i)
    #print(disparityMap)
    
#img[y_cord][x_cord]  
def getDisparityMap(features1,features2,img1,img2,windowSize):
    disparityMap = []
    for i in range(0,len(img1)):
        disparityMap = disparityMap + [np.zeros(len(img1[i]))]
        

    #print(len(disparityMap[0]))    
    for i in features1:
        maxCC = 0
        matchingCords = [-1,-1]
        for j in features2:
            if abs(j[1]-i[1]) < 11:
                   newCC = getCC(img1,img2,windowSize,i[0],i[1],j[0],j[1])
                   if newCC > maxCC:
                       maxCC = newCC
                       matchingCords = [j[1],j[0]]
        dx = abs(i[0]-matchingCords[0])
        #print("dx is " + str(dx))
        
        for  k in range(i[1],i[1] + windowSize,1):
           # print("in k loop")
            for m in range(i[0],i[0] + windowSize,1):
                disparityMap[k][m] = dx
                None
                
    return disparityMap
       
#Here img is a 2d array   
    
#returns top left pixel of regions
def findInterestRegions(img,windowSize):
    threshold = 50000
    regions = []
    #represents grid read left to right top to bot
    I0,I1,I2,I3,I4,I5,I6,I7 = 0,0,0,0,0,0,0,0 
    I = 0
    x,y = windowSize,windowSize
    while(True):
        I0,I1,I2,I3,I4,I5,I6,I7 = 0,0,0,0,0,0,0,0 
        I = 0
        if (x+windowSize*2 >= len(img[y])):
            x = 0
            y+=windowSize
        if (y+windowSize*2 >= len(img)):
            break
        #print("x is " + str(x))
        #print("y is " + str(y))
        I0 = getInterest(img,windowSize,x-windowSize,y-windowSize)
        I1 = getInterest(img,windowSize,x,y-windowSize)
        I2 = getInterest(img,windowSize,x+windowSize,y-windowSize)
        I3 = getInterest(img,windowSize,x-windowSize,y)
        I = getInterest(img,windowSize,x,y)
        I4 = getInterest(img,windowSize,x+windowSize,y)
        I5 = getInterest(img,windowSize,x-windowSize,y+windowSize)
        I6 = getInterest(img,windowSize,x,y+windowSize)
        I7 = getInterest(img,windowSize,x+windowSize,y+windowSize)
        #print("i is" + str(I))
        if I == max(I0,I1,I2,I3,I4,I5,I6,I7,I):     
            if I >= threshold:
                #regions+=[(x+windowSize)/2,(y+windowSize)/2]
                regions+=[[x,y]]
            
        x = x+windowSize
    #print(regions)        
    
    return regions

def getInterest(img,windowSize, x,y):
    I1,I2,I3,I4 = 0,0,0,0
    for j in range(x,x+windowSize):
            for i in range(y,y+windowSize):
                I1 = I1 + (img[i][j] - img[i][j+1])**2
                I2 = I2 + (img[i][j] - img[i+1][j])**2
                I3 = I3 + (img[i][j] - img[i+1][j+1])**2
                I4 = I4 + (img[i][j] - img[i+1][j-1])**2
    I = min(I1,I2,I3,I4)
     
    return I

def getCC(imgL,imgR,windowSize, xL,yL,xR,yR):
    CC = -1
    sumNumerator = 0
    sumDenomenator  = 0
    sumF1 = 0
    sumF2 = 0
    f1Avg = getIntensityAvg(imgL,windowSize,xL,yL)
    f2Avg = getIntensityAvg(imgR,windowSize,xR,yR)
    for i in range(0,windowSize):
        
        for j in range(0,windowSize):
            sumF1 += (imgL[yL+i][xL+j]-f1Avg)**2
            sumF2 += (imgR[yR+i][xR+j]-f2Avg)**2
            temp = (imgL[yL+i][xL+j]-f1Avg)*(imgR[yR+i][xR+i]-f2Avg)

            sumNumerator += temp
    sumDenomenator = np.sqrt(sumF1*sumF2)
    CC = sumNumerator/sumDenomenator
    #print(type(sumNumerator))
    #print("CC is " + str(CC))
    
    
    return CC

def getIntensityAvg(img,windowSize,x,y):
    avg = 0
   #numPixels = len(img)*len(img[0])
    
    #print("pixel intensity is " + str(img[x][y]))
    numPixels = windowSize*windowSize
    for i in range(x,x+windowSize):
        for j in range(y,y+windowSize):
            avg = avg + img[j][i]
    avg = avg/numPixels    
    return avg

main()
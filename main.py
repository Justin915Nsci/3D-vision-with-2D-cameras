import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import csv


def main():
    #imgL = cv2.imread('pics/im0.png',1)
    #imgR = cv2.imread('pics/im1.png',1)
    imgL = cv2.imread('pics/im0.jpg',1)
    imgR = cv2.imread('pics/im1.jpg',1)
   
    imgL_gry = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gry = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgL_gry',imgL_gry)
    #cv2.waitKey(0)
    windowSize = 7
    try:
        f = open("lFeats.txt")
        print("scanning lFeats.txt")
        data = csv.reader(f, delimiter = ',')
        #print("sucessful csv scan for lFeats")
        leftFeatures= []
        cnt = 0
        for row in data:
            if cnt%100 == 0:
                print(cnt)
            temp = []
            for i in row:
                try:
                    temp = temp + [int(i)]
                except:
                    None
            leftFeatures = leftFeatures + [temp]
            cnt = cnt + 1
        #print(leftFeatures)
        f.close()
    except:
        print("lFeats.txt not found, saving new leftFeatures data.")
        leftFeatures = findInterestRegions(imgL_gry,windowSize)
        f = open("lFeats.txt","w")
       
        for i in leftFeatures:
            for j in i:
                f.write(str(j))
                f.write(",")
            f.write("\n")
            
        f.close()
    
            
    try:
        f = open("rFeats.txt")
        print("scanning rFeats.txt")
        data = csv.reader(f, delimiter = ',')
        rightFeatures= []
        cnt = 0
        for row in data:
            if cnt%100 == 0:
                print(cnt)
            temp = []
            for i in row:
                try:
                    temp = temp + [int(i)]
                except:
                    None
            rightFeatures = rightFeatures + [temp]
            cnt= cnt +1
        #print(rightFeatures)
        f.close()
    except:
        print("rFeats.txt not found, saving new leftFeatures data.")
        rightFeatures = findInterestRegions(imgR_gry,windowSize)
        f = open("rFeats.txt","w")
        for i in rightFeatures:
            for j in i:
                f.write(str(j))
                f.write(",")
            f.write("\n")
        f.close()
    
    if len(leftFeatures) < len(rightFeatures):
        tImgGry = imgL_gry
        tImgRGB = imgL
        cImgGry = imgR_gry
        cImgRGB = imgR
        tFeats = leftFeatures
        cFeats = rightFeatures
    else:
        cImgGry = imgL_gry
        cImgRGB = imgL
        tImgGry = imgR_gry
        tImgRGB = imgR
        tFeats = rightFeatures
        cFeats = leftFeatures
    
    b = 193.001
    f = 3979.911
    dOff = 124.343
    
    maxD = 0
    try:
        f = open("purposeFail")
        f = open("depthMap.txt")
        #print("scanning depthMap.txt")
        data = csv.reader(f, delimiter = ',')
        disparityMap = []
        for row in data:
            temp = []
            for i in row:
                try:
                    temp = temp + [float(i)]
                    if float(i) >maxD:
                        maxD = float(i)
                except:
                    None
            disparityMap = disparityMap + [temp]
        #print(rightFeatures)
        f.close()
        
    except:
        print("depthMap.txt not found, saving new depthMap data.")
        disparityMap = getDisparityMap(tFeats,cFeats,tImgGry,cImgGry,windowSize,tImgRGB)
        for i in range(0,len(disparityMap)):
            for j in range(0,len(disparityMap[i])):
                if disparityMap[i][j] != 0:
                    #print("prev:" + str(j))
                    disparityMap[i][j] = b*f/(disparityMap[i][j]+dOff)
                    if disparityMap[i][j] >maxD:
                        maxD = disparityMap[i][j]
                    #print("new:" + str(j))
     
        f = open("depthMap.txt","w")
        for i in disparityMap:
            for j in i:
                f.write(str(j))
                f.write(",")
            f.write("\n")
        f.close()
    

    print("Scanning finished")
    print("generating image")
    #print(disparityMap)
    dMap = np.asarray(disparityMap,dtype=np.float32)
    bL,gL,rL = cv2.split(imgL)
    imgL_rgb = cv2.merge((rL,gL,bL))
    #bR,gR,rR = cv2.split(imgR)
    #imgR_rgb = cv2.merge((rR,gR,bR))
    
    #xx, yy = np.mgrid[0:imgL_rgb.shape[0],0:imgL_rgb.shape[1]]   
    fig = plt.figure()   
    ax = plt.axes(projection='3d')
    ax.set_ylabel('Z axis')
    ax.set_xlabel('X axis')
    ax.set_zlabel('Y axis')
    ax.set_title("Bicycle in 3D")
    #imageArray = np.asarray(imgL_gry)
    #distanceArray = np.vectorize(disp)
    for col in range(0,len(dMap)):
        for row in range(0,len(dMap[0])):
            dist = dMap[col][row]
            r,g,b = tImgRGB[col][row]
            if dist == 0:
                d = 0
            else:
                d = maxD-dist
            plt.plot([len(dMap) - row],[d],[len(dMap[0]) - col],marker = "o", color=(b/255,g/255,r/255), markersize = 0.5)
    

    
    while True:
        azim = int(input("axim:"))
        elev = int(input("elev:"))
        ax.view_init(azim=azim,elev = elev)
        
        plt.savefig(str(azim) + "_" + str(elev))
        cont = input("continue?:")
        if cont == "no":
            break
    
#img[y_cord][x_cord]  
def getDisparityMap(features1,features2,img1,img2,windowSize,tImgRGB):
    disparityMap = []
    for i in range(0,len(img1)):
        disparityMap = disparityMap + [np.zeros(len(img1[i]))]
        
    
    #print(len(disparityMap[0]))    
    cnt = 0
    for i in features1:
        stack = []
        #if cnt%100 == 0:
        if cnt%1 == 0:
            print(str(len(features1)- cnt) + " features left to analyze")
        maxCC = 0
        mC = [-1,-1] #stores coordinates for matching conjugate pair
        
        h = len(img1)
        w = len(img1[0])
        i1Matrix = np.zeros((h,w))
        f1 = 0
        for k in range(i[0],i[0]+windowSize):
            if i1Matrix[i[1]][k] == 0:
                i1Matrix[i[1]][k] = getIntensityCol(img1, windowSize, k, i[1])
            f1+=i1Matrix[i[1]][k]
        f1 = f1/windowSize
        
        for j in features2:
            if abs(j[1]-i[1]) < 4:
                   newCC = getCC(img1,img2,windowSize,i[0],i[1],j[0],j[1],f1)
                   if newCC > maxCC:
                       maxCC = newCC
                       mC = [j[0],j[1]]
        dx = abs(i[0]-mC[0])
        #print("mC is" + str(mC))
        
        disparityMap[i[1]][i[0]] = dx
        
        r,g,b = tImgRGB[i[1]][i[0]]
        #print("color OG:" + str(r) +"," + str(g) + "," + str(b))
        #fillDepthMap(disparityMap,tImgRGB,i[0]-1,i[1]-1,r,g,b,dx)
        #fillDepthMap(disparityMap,tImgRGB,i[0],i[1]-1,r,g,b,dx)
        #fillDepthMap(disparityMap,tImgRGB,i[0]+1,i[1]-1,r,g,b,dx)
        #fillDepthMap(disparityMap,tImgRGB,i[0]-1,i[1],r,g,b,dx)
        #fillDepthMap(disparityMap,tImgRGB,i[0]+1,i[1],r,g,b,dx)
        #fillDepthMap(disparityMap,tImgRGB,i[0]-1,i[1]+1,r,g,b,dx)
        #fillDepthMap(disparityMap,tImgRGB,i[0],i[1]+1,r,g,b,dx)
        #fillDepthMap(disparityMap,tImgRGB,i[0]+1,i[1]+1,r,g,b,dx)
        stack.append([i[0]-1,i[1]-1])
        stack.append([i[0],i[1]-1])
        stack.append([i[0]+1,i[1]-1])
        stack.append([i[0]-1,i[1]])
        stack.append([i[0]+1,i[1]])
        stack.append([i[0]-1,i[1]+1])
        stack.append([i[0],i[1]+1])
        stack.append([i[0]+1,i[1]+1])
        while True:
            if len(stack) == 0:
                break
            cords = stack.pop()
            fillDepthMap(disparityMap,tImgRGB,cords[0],cords[1],r,g,b,dx,stack)
            #print("stack size:" + str(len(stack)))
        
        cnt = cnt +1
                
    return disparityMap
     
#apply this function to all pixels around interest point
def fillDepthMap(dMap,img,x,y,r,g,b,d,stack):
    #print("current cords:" + str(x) + "," + str(y))
    #print("d is " + str(d))
    if d < 20:
        return False
    threshold = 120
	#check to see if these x and y coords are worth analyzing
    valid = False
    try:
        if(dMap[y][x]==0):
			#print("is valid")
            valid = True
    except:
        print(str(x) + "," + str(y) + "are invalid coordinates")

    if valid ==False:
        #print("depth at " + str(x) + "," + str(y) + " is already found")
        return False
    
    
    imgR,imgG,imgB = img[y][x]
    tally = 0
    if abs(imgR-r)>threshold:
        tally = tally + 1
    if abs(imgB-b)>threshold:
        tally = tally + 1    
    if abs(imgG-g)>threshold:
        tally = tally + 1
    if tally >0:
        return False
        
    #if(abs(imgR-r)>threshold) or (abs(imgG-g)>threshold) or (abs(imgB-b)>threshold):
        #print([abs(imgR-r),abs(imgG-g),abs(imgB-b)])
     #   print("color here:" + str(imgR) +"," + str(imgG) + "," + str(imgB))
      #  print("color should be:" + str(r) +"," + str(g) + "," + str(b))
       # valid = False
        #return False

    else:
        dMap[y][x] = d
    if valid == False:
        return False
    else:

        stack.append([x-1,y-1])
        stack.append([x,y-1])

        stack.append([x+1,y-1])
     
        stack.append([x-1,y])
        stack.append([x+1,y])
      
        stack.append([x-1,y+1]) 

        stack.append([x,y+1])

        stack.append([x+1,y+1])
        
    return True




#Here img is a 2d array   
    
#returns top left pixel of regions
def findInterestRegions(img,windowSize):
    print("Scanning Regions")
    threshold = 1500000
    regions = []
    #represents grid read left to right top to bot
    I0,I1,I2,I3,I4,I5,I6,I7 = 0,0,0,0,0,0,0,0 
    I = 0
    x,y = windowSize,windowSize
    h = len(img)
    w = len(img[0])
    iArray = np.zeros((h,w))
    while(True):
        shiftDown = False
        #I0,I1,I2,I3,I4,I5,I6,I7 = 0,0,0,0,0,0,0,0 
        #I = 0
        if (x+windowSize*2 >= len(img[y])):
            x = 0
            y+=1
            print("y is " +  str(y))
            shiftDown = True
        if (y+windowSize*2 >= len(img)):
            break
        #print("x is " + str(x))
        #print("y is " + str(y))
        I0 = iArray[y-1][x-1]
        I1 = iArray[y-1][x]
        I2 = iArray[y-1][x+1]
        I3 = iArray[y][x-1]
        I = iArray[y][x]
        I4 = iArray[y-1][x+1]
        I5 = iArray[y-1][x]
        I6 = iArray[y-1][x]
        I7 = iArray[y-1][x]
        if I == 0:    
            I = getInterest(img,windowSize,x,y)
            iArray[y][x] = I
        #print("I is " + str(I))
        
        if I<=threshold:
            None
        else:      
            if I0 == 0:
                I0 = getInterest(img,windowSize,x-1,y-1)
                iArray[y-1][x-1] = I0
            
            if I1 == 0:
                I1 = getInterest(img,windowSize,x,y-1)
                iArray[y-1][x] = I1
            
            if I2 == 0:
                I2 = getInterest(img,windowSize,x+1,y-1)
                iArray[y-1][x+1] = I2
            
            if I3 == 0:    
                I3 = getInterest(img,windowSize,x-1,y)
                iArray[y][x-1] = I3
            
            if I4 == 0:    
                I4 = getInterest(img,windowSize,x+1,y)
                iArray[y][x+1] = I4
        
            if I5 == 0:
                I5 = getInterest(img,windowSize,x-1,y+1)
                iArray[y+1][x-1] = I5
       
            if I6 == 0:
                I6 = getInterest(img,windowSize,x,y+1)
                iArray[y+1][x] = I6
            
            if I7 == 0:
                I7 = getInterest(img,windowSize,x+1,y+1)
                iArray[y+1][x+1] = I7
                #print("i is" + str(I))
            if I == max(I0,I1,I2,I3,I4,I5,I6,I7,I):     
                regions+=[[x,y]]
            
        x = x+1       
    regions.reverse()
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

def getCC(imgL,imgR,windowSize, xL,yL,xR,yR,f1):
    CC = -1
    sumNumerator = 0
    sumDenomenator  = 0
    sumF1 = 0
    sumF2 = 0
    h = len(imgL)
    w = len(imgL[0])
    i1Matrix = np.zeros((h,w))
    i2Matrix = np.zeros((h,w))
    if f1==None:
        f1 = 0
        for i in range(xL,xL+windowSize):
            if i1Matrix[yL][i] == 0:
                i1Matrix[yL][i] = getIntensityCol(imgL, windowSize, i, yL)
            f1+=i1Matrix[yL][i]
        f1 = f1/windowSize
        f1Avg = f1
            
        #f1Avg = getIntensityAvg(imgL,windowSize,xL,yL)
    else:
        f1Avg = f1
        
    
    f2Avg= 0
    for i in range(xR,xR+windowSize):
        if i2Matrix[yR][i] == 0:
                i2Matrix[yR][i] = getIntensityCol(imgR, windowSize,i,yR)
        f2Avg += i2Matrix[yR][i]
    f2Avg = f2Avg/windowSize
    
    #f2Avg = getIntensityAvg(imgR,windowSize,xR,yR)
    for i in range(0,windowSize):
        
        for j in range(0,windowSize):
            sumF1 += (imgL[yL+i][xL+j]-f1Avg)**2
            sumF2 += (imgR[yR+i][xR+j]-f2Avg)**2
            temp = (imgL[yL+i][xL+j]-f1Avg)*(imgR[yR+i][xR+j]-f2Avg)

            sumNumerator += temp
    sumDenomenator = np.sqrt(sumF1*sumF2)
    CC = sumNumerator/sumDenomenator
    
    
    return CC

def getIntensityCol(img,windowSize,x,y):
    avg = 0
    numPixels = windowSize
    for j in range(y,y+windowSize):
        avg = avg + img[j][x]
    avg = avg/numPixels
    return avg

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

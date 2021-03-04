#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 06:52:43 2021

@author: tatianamiti

"""

import os,json

import numpy as np
#from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix



import csv
#from shapely.geometry import Polygon


########## loading all data

scalefactor = 0.5022
side = 200
areaCuttOff = 0.5 


if os.path.isfile('PolyList.txt'):
    with open('PolyList.txt', 'r') as f_f:
        PolyList = json.loads(f_f.read())
else:
    PolyList = []
    
if len(PolyList)==0:   
    myListx = []
    myListy = []
    myL1 = []
    myL2 = []

    with open('Polygons.csv', newline='') as csvfile:
       reader = csv.DictReader(csvfile)
       for row in reader:
           myListx.append(int(row['X']))
           myListy.append(int(row['Y']))
           myL1.append(int(row['L1']))
           myL2.append(int(row['L2']))

     
        #### dividing data in to Stroma+ Positive and Stroma + /negative  for plotting
    AllList = []
    keyListTemp = []
    KeyList = []
    PolyList = []
    for i in range(len(myListx)):
        AllList.append([(myListx[i]*scalefactor,myListy[i]*scalefactor), (myL1[i],myL2[i])])
     
    for j in range(len(myListx)):
        keyListTemp.append(AllList[j][1])
        
    setList = set(keyListTemp)
    KeyList = list(setList)
    #print(KeyList, len(KeyList))
        
    PolyDict = {}    
    for value in KeyList:
        tempList = []
        for entr in AllList:
            if entr[1] == value:
                tempList.append(entr[0])
        PolyDict[value] = tempList[:]
    
    tempList = []
    for key in PolyDict:
        PolyList.append(PolyDict[key])
    
        
    
        # if os.path.isfile('AllList.txt'):
        #       with open('AllList.txt', 'w') as ff_f:
        #           ff_f.write(json.dumps(AllList))
        # else:
        #       with open('AllList.txt', 'w') as f_ff:
        #           f_ff.write(json.dumps(AllList))
        
        if os.path.isfile('PolyList.txt'):
              with open('PolyList.txt', 'w') as f_f:
                  f_f.write(json.dumps(PolyList))
        else:
            with open('PolyList.txt', 'w') as f_f:
                  f_f.write(json.dumps(PolyList))
                  
                  
        

from planar import Polygon
innerPolyList = []
for f in PolyList:
    for d in PolyList:
        if d != f:
            polyg = Polygon(d)
            if polyg.contains_point(f[0]):
                innerPolyList.append(f)
#print(innerPolyList)
#Get alist of External Polygons only
PolyListNoInner = []
for dd in PolyList:
    if dd not in innerPolyList:
        PolyListNoInner.append(dd)
        
print("len polylistNoInner", len(PolyListNoInner))

from shapely.geometry import Polygon
# get a list on polygon areas and the index in its ExternPoly
polyArea = []
for jj in range(len(PolyListNoInner)):
    myPoly = Polygon(PolyListNoInner[jj])
    polyArea.append([myPoly.area, jj])
   
    

    
#polyArea.sort()  
# select only the largest 20 polygons, since the rest don't matter!!
PolyListCutOff = []
for ii in range(len(polyArea)):
    if polyArea[ii][0] > side*side:
    #print(polyArea[ii][1])
        PolyListCutOff.append(PolyListNoInner[polyArea[ii][1]])
    

PolyDict = {}
PolyList = []
PolyListNoInner = []
polyArea = []
AllList = []
myListx = []
myListy = []
myL1 = []
myL2 = []
keyListTemp = []
KeyList = []
setList = []

#print('larg epolygons')    
#print(len(largePoly))
    
# plt.figure()
# for i in PolyListCutOff:
#     polygon1 = Polygon(i)
#     x,y = polygon1.exterior.xy
#     plt.plot(x,y)
#     #plt.plot(polygon1)
# plt.show()


# load the files to populate the qudrants

if os.path.isfile('StromaList.txt'):
    with open('StromaList.txt', 'r') as fff:
        StromaList = json.loads(fff.read())
else:
    StromaList = []
    

if os.path.isfile('NegativeList.txt'):
        with open('NegativeList.txt', 'r') as f_ff:
            NegativeList = json.loads(f_ff.read())
else:
            NegativeList = []
            
            
if os.path.isfile('PositiveList.txt'):
        with open('PositiveList.txt', 'r') as f_f:
            PositiveList = json.loads(f_f.read())
else:
            PositiveList = []

#print('PolyListCutOff[0]')
print(len(PolyListCutOff))
from planar import Polygon
#pointsfullPoly = []
for tu in range(len(PolyListCutOff)):
    plt.close("all")
    brduPos = [] #post [0]
    brduNeg = [] #pos [1]
    Stroma = [] #pos [2]
    #polyall = []
    for gh in PositiveList:
        pt = (gh[0],gh[1])
        if Polygon(PolyListCutOff[tu]).contains_point(pt):
            #print("pos")
            brduPos.append([gh[0],gh[1]])
            #pointsfullPoly.append(gh)
    for df in NegativeList:
        pt1 = (df[0], df[1])
        if Polygon(PolyListCutOff[tu]).contains_point(pt1):
            #print("neg")
            brduNeg.append([df[0],df[1]])
            #pointsfullPoly.append(df)
    for hj in StromaList:
        pt2 = (hj[0],hj[1])
        if Polygon(PolyListCutOff[tu]).contains_point(pt2):
            Stroma.append([hj[0],hj[1]])
            #pointsfullPoly.append(hj)
    #polyall.append([brduPos,brduNeg,Stroma])
            
#print("did the points")
#print(len(brduPos), len(brduNeg), len(Stroma))



### loading initial data
#=======================================================================================================
    if os.path.isfile('StromaStromaList'+ str(tu)+'.txt'):
        with open('StromaStromaList'+str(tu)+'.txt', 'r') as fff1:
            StromaStromaList = json.loads(fff1.read())
    else:
        StromaStromaList = []
        
    
    
    if len(StromaStromaList)==0:
        # if os.path.isfile('StromaList.txt'):
        #     with open('StromaList.txt', 'r') as fff:
        #         StromaList = json.loads(fff.read())
        # else:
        #     StromaList = []
               
        # print("stroma", len(StromaList))
        # tmpStromaArray = []
        
        # for i in range(len(StromaList)):
        #     tmpStromaArray.append([int(StromaList[i][0]), int(StromaList[i][1])])        
                
        StromaArray = np.array(Stroma)
       
        StromaStromaList = distance_matrix(StromaArray, StromaArray)   
        StromaStromaList = StromaStromaList.flatten()
        StromaStromaList = StromaStromaList[StromaStromaList <= 300]
        #StromaStromaList = np.unique(StromaStromaList)
        #max_stromaStroma = np.amax(StromaStromaList)
        #StromaStromaList = (np.sqrt(2)*(1.0/max_stromaStroma))*StromaStromaList
        #print(len(StromaStromaList))
    
        plt.figure()
        plt.hist(StromaStromaList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'StromaStroma_' + str(tu))   
        plt.xlabel(r'$Distance\ from\ Stroma\ to\ Stroma\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig("StromaStroma_"+ str(tu) + ".png")
        
        #StromaStromaList = pd.Series(StromaStromaList).to_json(orient='values')
        if os.path.isfile('StromastromaList' + str(tu) +'.txt'):
                with open('StromaStromaList' + str(tu) + '.txt', 'w') as fff1:
                    fff1.write(json.dumps(StromaStromaList.tolist()))
        else:
                with open('StromaStromaList'+ str(tu) +'.txt', 'w') as fff1:
                    fff1.write(json.dumps(StromaStromaList.tolist()))
                    
    else:
        plt.figure()
        plt.hist(StromaStromaList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'StromaStroma_' + str(tu))   
        plt.xlabel(r'$Distance\ from\ Stroma\ to\ Stroma\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('StromaStroma_'+str(tu)+'.png')
    
    StromaStromaList = []
    
    #=======================================================================================================
    
    if os.path.isfile('PositivePositiveList'+str(tu)+'.txt'):
        with open('PositivePositiveList'+str(tu)+'.txt', 'r') as f_f1:
            PositivePositiveList = json.loads(f_f1.read())
    else:
            PositivePositiveList = []  
            
    if len(PositivePositiveList) ==0:
    
        # if os.path.isfile('PositiveList.txt'):
        #     with open('PositiveList.txt', 'r') as f_f:
        #         PositiveList = json.loads(f_f.read())
        # else:
        #         PositiveList = []
        
        # tmpPositiveArray = []
        
        # for j in range(len(PositiveList)):
        #     tmpPositiveArray.append([int(PositiveList[j][0]), int(PositiveList[j][1])])  
            
        PositiveArray = np.array(brduPos)
                
        PositivePositiveList = distance_matrix(PositiveArray, PositiveArray)   
        PositivePositiveList = PositivePositiveList.flatten()
        PositivePositiveList = PositivePositiveList[PositivePositiveList <= 300]
        #PositivePositiveList = np.unique(PositivePositiveList)
        #max_positivePositive = np.amax(PositivePositiveList)
        #PositivePositiveList= (np.sqrt(2)*(1.0/max_positivePositive))*PositivePositiveList
        #print(PositivePositiveList[6], max_positivePositive)
         
        plt.figure()
        plt.hist(PositivePositiveList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'PositivePositive_'+ str(tu))   
        plt.xlabel(r'$Distance\ from\ Positive\ to\ Positive\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('PositivePositive'+str(tu)+'.png')       
        
        #PositivePositiveList = pd.Series(PositivePositiveList).to_json(orient='values')
        if os.path.isfile('PositivePositiveList'+str(tu)+'.txt'):
                with open('PositivePositiveList'+str(tu)+'.txt', 'w') as f_f1:
                    f_f1.write(json.dumps(PositivePositiveList.tolist()))
        else:
                with open('PositivePositiveList'+str(tu)+'.txt', 'w') as f_f1:
                    f_f1.write(json.dumps(PositivePositiveList.tolist()))
                    
    else:
        plt.figure()
        plt.hist(PositivePositiveList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'PositivePositive_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Positive\ to\ Positive\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('PositivePositive'+str(tu)+'.png')  
        
    PositivePositiveList = []
            
    #========================================================================================
    if os.path.isfile('NegativeNegativeList'+str(tu)+'.txt'):
        with open('NegativeNegativeList'+str(tu)+'.txt', 'r') as f_ff1:
            NegativeNegativeList = json.loads(f_ff1.read())
    else:
            NegativeNegativeList = [] 
    
    if len(NegativeNegativeList) == 0:
    
        # if os.path.isfile('NegativeList.txt'):
        #     with open('NegativeList.txt', 'r') as f_ff:
        #         NegativeList = json.loads(f_ff.read())
        # else:
        #         NegativeList = []
        # print(len(NegativeList))
                
        # tmpNegativeArray = []
                
        # for k in range(len(NegativeList)):
        #     tmpNegativeArray.append([int(NegativeList[k][0]), int(NegativeList[k][1])])         
        
        NegativeArray = np.array(brduNeg)
        
        #print(len(NegativeNegativeList))        
        #NegativeNegativeList = distance_matrix(NegativeArray[:int(len(NegativeArray))], NegativeArray[-int(len(NegativeArray)):])   
        NegativeNegativeList = distance_matrix(NegativeArray, NegativeArray)   
        NegativeNegativeList = NegativeNegativeList.flatten()
        NegativeNegativeList = NegativeNegativeList[NegativeNegativeList <= 300]
        #NegativeNegativeList = np.unique(NegativeNegativeList)
        #max_negativeNegative = np.amax(NegativeNegativeList)
        #NegativeNegativeList= (np.sqrt(2)*(1.0/max_negativeNegative))*NegativeNegativeList
        
        plt.figure()
        plt.hist(NegativeNegativeList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'NegativeNegative_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Negative\ to\ Negative\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('NegativeNegative'+str(tu) +'.png')       
        
        #NegativeNegativeList = pd.Series(NegativeNegativeList).to_json(orient='values')
        
        if os.path.isfile('NegativeNegativeList'+str(tu)+'.txt'):
                with open('NegativeNegativeList'+str(tu)+'.txt', 'w') as f_ff1:
                    f_ff1.write(json.dumps(NegativeNegativeList.tolist()))
        else:
                with open('NegativeNegativeList'+str(tu)+'.txt', 'w') as f_ff1:
                    f_ff1.write(json.dumps(NegativeNegativeList.tolist()))
                    
    else:
        plt.figure()
        plt.hist(NegativeNegativeList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'NegativeNegative_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Negative\ to\ Negative\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('NegativeNegative'+str(tu)+'.png') 
    
    NegativeNegativeList = []
             
    #=================================================================================================        
     
    if os.path.isfile('PositiveNegativeList.txt'):
        with open('PositiveNegativeList.txt', 'r') as f_ff3:
            PositiveNegativeList = json.loads(f_ff3.read())
    else:
        PositiveNegativeList = []
    
    if len(PositiveNegativeList) == 0:
    
        PositiveNegativeList = distance_matrix(NegativeArray, PositiveArray)   
        PositiveNegativeList = PositiveNegativeList.flatten()
        PositiveNegativeList = PositiveNegativeList[PositiveNegativeList <= 300]
        #PositiveNegativeList = np.unique(PositiveNegativeList)
        #max_positiveNegative = np.amax(PositiveNegativeList)
        #PositiveNegativeList= (np.sqrt(2)*(1.0/max_positiveNegative))*PositiveNegativeList
        
        plt.figure()
        plt.hist(PositiveNegativeList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'PositiveNegative_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Negative\ to\ Positive\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('PositiveNegative'+str(tu)+'.png')       
        
        #PositiveNegativeList = pd.Series(PositiveNegativeList).to_json(orient='values')
        
        if os.path.isfile('PositiveNegativeList'+str(tu)+'.txt'):
                with open('PositiveNegativeList'+str(tu)+'.txt', 'w') as f_ff3:
                    f_ff3.write(json.dumps(PositiveNegativeList.tolist()))
        else:
                with open('PositiveNegativeList'+str(tu)+'.txt', 'w') as f_ff3:
                    f_ff3.write(json.dumps(PositiveNegativeList.tolist()))
                    
    else:
        plt.figure()
        plt.hist(PositiveNegativeList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'PositiveNegative_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Negative\ to\ Positive\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('PositiveNegative'+str(tu)+'.png')     
    
    PositiveNegativeList = []
    
    #========================================================================================================
    
    if os.path.isfile('StromaPositiveList'+str(tu)+'.txt'):
        with open('StromaPositiveList'+str(tu)+'.txt', 'r') as f_ff4:
            StromaPositiveList = json.loads(f_ff4.read())
    else:
            StromaPositiveList = []
            
    if len(StromaPositiveList) ==0:
        StromaPositiveList = distance_matrix(StromaArray, PositiveArray)   
        StromaPositiveList = StromaPositiveList.flatten()
        StromaPositiveList = StromaPositiveList[StromaPositiveList <= 300]
        #StromaPositiveList = np.unique(StromaPositiveList)
        #max_stromaPositive = np.amax(StromaPositiveList)
        #StromaPositiveList= (np.sqrt(2)*(1.0/max_stromaPositive))*StromaPositiveList
        
        plt.figure()
        plt.hist(StromaPositiveList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'StromaPositive_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Stroma\ to\ Positive\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('StromaPositive'+str(tu)+'.png')       
        
        #StromaPositiveList = pd.Series(StromaPositiveList).to_json(orient='values')
        
        if os.path.isfile('StromaPositiveList'+str(tu)+'.txt'):
                with open('StromaPositiveList'+str(tu)+'.txt', 'w') as f_ff4:
                    f_ff4.write(json.dumps(StromaPositiveList.tolist()))
        else:
                with open('StromaPositiveList'+str(tu)+'.txt', 'w') as f_ff4:
                    f_ff4.write(json.dumps(StromaPositiveList.tolist()))
                    
    else:
        plt.figure()
        plt.hist(StromaPositiveList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'StromaPositive_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Stroma\ to\ Positive\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('StromaPositive'+str(tu)+'.png')     
    
    StromaPositiveList = []
    #=====================================================================================================================
    
    if os.path.isfile('StromaNegativeList'+str(tu)+'.txt'):
        with open('StromaNegativeList'+str(tu)+'.txt', 'r') as f_ff2:
            StromaNegativeList = json.loads(f_ff2.read())
    else:
            StromaNegativeList = []
    
    
    if len(StromaNegativeList)==0:
        StromaNegativeList = distance_matrix(StromaArray, NegativeArray)   
        StromaNegativeList = StromaNegativeList.flatten()
        StromaNegativeList = StromaNegativeList[StromaNegativeList <= 300]
        #StromaNegativeList = np.unique(StromaNegativeList)
        #max_stromaNegative = np.amax(StromaNegativeList)
        #StromaNegativeList= (np.sqrt(2)*(1.0/max_stromaNegative))*StromaNegativeList
        
        plt.figure()
        plt.hist(StromaNegativeList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'StromaNegative_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Stroma\ to\ Negative\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('StromaNegative'+str(tu)+'.png')       
        
        #StromaNegativeList = pd.Series(StromaNegativeList).to_json(orient='values')
        
        if os.path.isfile('StromaNegativeList'+str(tu)+'.txt'):
                with open('StromaNegativeList'+str(tu)+'.txt', 'w') as f_ff3:
                    f_ff3.write(json.dumps(StromaNegativeList.tolist()))
        else:
                with open('StromaNegativeList.txt', 'w') as f_ff3:
                    f_ff3.write(json.dumps(StromaNegativeList.tolist()))
                    
    else:
        plt.figure()
        plt.hist(StromaNegativeList,bins = 200,density=True,color = 'blue', alpha=0.6,label = 'StromaNegative_'+str(tu))   
        plt.xlabel(r'$Distance\ from\ Stroma\ to\ Negative\ (um)$', fontsize = 14)
        plt.ylabel(r'$Frequency$', fontsize = 14)
        plt.legend(loc= 'upper right')
        plt.savefig('StromaNegative'+str(tu)+'.png')   
    
    StromaNegativeList = []
    # =============================================================================







#plt.hist(RandStrList,bins = 500,density=True,color = 'blue', alpha=0.9,label = 'Random')   
# #plt.hist(alldistPos,bins = 500,density=True,color = 'orange', alpha=0.4,label = 'BrdU Positive')   
# plt.hist(alldistNeg,bins = 500,density=True,color = 'red', alpha =0.4,label = 'BrdU Negative')   
# plt.xlabel(r'$Distance\ from\ Stroma\ (um)$', fontsize = 14)
# plt.ylabel(r'$Cell\ Number$', fontsize = 14)
# plt.legend(loc= 'upper right')
# plt.savefig('randomneg.jpeg')
# plt.show()

# StrTemp= [i for i in RandStrList if i>13.0]
# RandStrList = []
# RandStrList = StrTemp
# from scipy import stats
# import seaborn as sns

# #sns.set_style('darkgrid')
# sns.distplot(RandStrList, color='red',label='Random')
# #sns.distplot(alldistPos,label = 'BrdU Positive')
# #sns.distplot(alldistNeg,label='BrdU Negative')
# plt.legend()
# plt.savefig('testPDFrandom')

# from numpy import asarray
# from numpy import exp
# from scipy import stats
# from scipy.stats import norm
# import numpy as np
# import scipy
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV

# sample =asarray(RandStrList)
# sample1 = asarray(alldistPos)
# sample2 = asarray(alldistNeg)
# model = KernelDensity(bandwidth=9, kernel='gaussian')
# sample = sample.reshape((len(RandStrList), 1))
# sample1 = sample1.reshape((len(alldistPos), 1))
# sample2 = sample2.reshape((len(alldistNeg), 1))

# model.fit(sample)
# values = asarray([value for value in range(1, 800)])
# values = values.reshape((len(values), 1))
# probabilities = model.score_samples(values)
# probabilities = exp(probabilities)
# # plot the histogram and pdf
# plt.hist(sample, bins=300, density=True, color = 'coral', label = "Random Data")
# plt.plot(values[:], probabilities, color='red',label = "Calculated PDF Random")
# plt.legend(loc='upper right')
# #plt.savefig('testPDFcalcrandomall')
# #plt.show()

# model.fit(sample1)
# values1 = asarray([value for value in range(1, 800)])
# values1 = values1.reshape((len(values1), 1))
# probabilities1 = model.score_samples(values1)
# probabilities1 = exp(probabilities1)
# # plot the histogram and pdf
# plt.hist(sample1, bins=300, density=True, color = 'green', alpha = 0.6, label = "BrdU+")
# plt.plot(values1[:], probabilities1, color='darkgreen',label = "Calculated PDF Positive")
# plt.legend(loc='upper right')

# model.fit(sample2)
# # sample probabilities for a range of outcomes
# values2 = asarray([value for value in range(1, 800)])
# values2 = values2.reshape((len(values2), 1))
# probabilities2 = model.score_samples(values2)
# probabilities2 = exp(probabilities2)
# plot the histogram and pdf
#plt.hist(sample, bins=100, density=True, color = 'silver', label = "Random Data")
#plt.hist(sample1, bins=100, density=True, color = 'orange', label = "positive Data")
# plt.hist(sample2, bins=300, density=True, alpha=0.5, color = 'royalblue', label = "BrdU-")
# plt.plot(values2[:], probabilities2, color='blue',label = "Calculated PDF Negatve")
# plt.legend(loc='upper right')
# plt.savefig('testPDFcalcrandomall')
# plt.show()

# img = 0
# sigma = 6.5
# colors = ['k' for i in range(len(point_rand))]
# def snapshot(pos, colors):

#     global img

#     pylab.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)

#     pylab.gcf().set_size_inches(20, 20)

#     pylab.axis([2.15e3, 2.3e4, 1.5e3, 1.3e04])

#     pylab.setp(pylab.gca(), xticks=[2.15e3, 2.3e4], yticks=[1.5e3, 1.3e04])
#     pylab.xticks(np.linspace(2.15e3, 2.3e4, 9, endpoint=True))
#     pylab.yticks(np.linspace(1.5e3, 1.3e04, 9, endpoint=True))

#     for (x, y), c in zip(pos, colors):

#         circle = pylab.Circle((x, y), radius=sigma, fc=c)

#         pylab.gca().add_patch(circle)
        
        
        
# #     hull = ConvexHull(AllListNoLabel)
# # #import matplotlib.pyplot as plt
# #     pylab.plot(AllListNoLabel[hull.vertices,0], AllListNoLabel[hull.vertices,1], 'r--', lw=2)
# #     pylab.plot(AllListNoLabel[hull.vertices[0],0], AllListNoLabel[hull.vertices[0],1], 'ro')
#     pylab.savefig('test_randPlot250k', transparent=True)
#     pylab.grid(True)
#     #pylab.show()
#     pylab.close()
#     img += 1

# snapshot(point_rand,colors)


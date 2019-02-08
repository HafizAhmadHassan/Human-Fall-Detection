import os
import copy
import numpy as np
from math import log, log10
from scipy import fftpack as ft
from collections import Counter
from scipy import stats, ndimage #stats.iqr, entropy

##########################################
def energy(a):
    return round(sum([x*x/len(a) for x in a]), 4)
##########################################
def entropy(x, logfun=lambda x: log(x, 2)):
    counts = Counter()
    total = 0.0
    for item in x:
        total += 1
        counts[item] += 1
    probs = [counts[k]/total for k in counts]
    ent = -sum([p*logfun(p) for p in probs])
    return ent
##########################################
def sma(X, Y, Z):
    sma = 0
    for i in range(0, len(X)):
        sma = sma + X[i] + Y[i] + Z[i]
    return round(sma/len(X), 4)
##########################################
def writeToFile(vec, outfile):
    for i in range(0, len(vec)):
        s = str(vec[i])
        outfile.write(s)
        outfile.write(' ')
##########################################
def readData(inputfilename):
    file = open(inputfilename)
    content = file.read()
    file.close()
    content = content.splitlines()
    Acc = [ [], [], [] ]    #X, Y, Z
    Gyro = [ [], [], [] ]   #X, Y, Z
    for r in range(0, len(content)):        #len(content) = 19999
        line = content[r]
        line = line.split(',')
        Acc[0].append(int(line[0])) #X
        Acc[1].append(int(line[1])) #Y
        Acc[2].append(int(line[2])) #Z
        Gyro[0].append(int(line[3])) #GyroX
        Gyro[1].append(int(line[4])) #GyroY
        Gyro[2].append(int(line[5])) #GyroZ
    return Acc, Gyro
##########################################
def createDictionary(alphas, betas):
    feats = dict.fromkeys(betas)
    col = {'X': 0, 'Y': 0, 'Z': 0}
    for key in feats.keys():
        feats[key] = copy.deepcopy(col)
    d = dict.fromkeys(alphas)
    for key in d.keys():
        d[key] = copy.deepcopy(feats)
    return d
##########################################
def getA(s, Acc, Gyro, dervAcc, dervGyro, row):    #row no of Acc or Gyro
    if s == 'tBodyAcc' or s == 'fBodyAcc':
        return [Acc[0][row+1] - Acc[0][row], Acc[1][row+1] - Acc[1][row], Acc[2][row+1] - Acc[2][row]]
    if s == 'tGravityAcc':
        return [9.81, 9.81, 9.81]
    if s == 'tBodyAccJerk' or s == 'fBodyAccJerk':
        return [dervAcc[0][row+1] - dervAcc[0][row], dervAcc[1][row+1] - dervAcc[1][row], dervAcc[2][row+1] - dervAcc[2][row]]
    if s == 'tBodyGyro' or s == 'fBodyGyro':
        return [ Gyro[0][row+1]-Gyro[0][row] , Gyro[1][row+1]-Gyro[1][row] , Gyro[2][row+1]-Gyro[2][row] ]
    if s == 'tBodyGyroJerk':
        return [ dervGyro[0][row+1]-dervGyro[0][row] , dervGyro[1][row+1]-dervGyro[1][row] , dervGyro[2][row+1]-dervGyro[2][row] ]
    if s == 'tBodyAccMag' or s == 'fBodyAccMag':
        return [ (Acc[0][row]**2 + Acc[1][row]**2 + Acc[2][row]**2)**0.5 ]
    if s == 'tGravityAccMag':
        return [ ( (9.81*3)**2 )**0.5 ]
    if s == 'tBodyAccJerkMag' or s == 'fBodyAccJerkMag':
        return [ (dervAcc[0][row]**2 + dervAcc[1][row]**2 + dervAcc[2][row]**2)**0.5 ]
    if s == 'tBodyGyroMag' or s == 'fBodyGyroMag':
        return [ (Gyro[0][row]**2 + Gyro[1][row]**2 + Gyro[2][row]**2)**0.5 ]
    if s == 'tBodyGyroJerkMag' or s == 'fBodyGyroJerkMag':
        return [ (dervGyro[0][row]**2 + dervGyro[1][row]**2 + dervGyro[2][row]**2)**0.5 ]

##########################################
def computeFeatures(alphas, betas, Acc, Gyro, dervAcc, dervGyro, row):
    d = createDictionary(alphas, betas)
    for key in d.keys():
        if key.rfind('Mag') == -1:
            Vec = []
            if key == 'tBodyAcc' or key == 'tGravityAcc':
                Vec = Acc
            elif key == 'tBodyGyro':
                Vec = Gyro
            elif key == 'tBodyAccJerk':
                Vec = dervAcc
            elif key == 'tBodyGyroJerk':
                Vec = dervGyro
            for c, k in enumerate(['X', 'Y', 'Z']):
                d[key]['mean'][k] = np.mean(Vec[c])
                d[key]['std'][k] = np.std(Vec[c])
                d[key]['med'][k] = np.median(Vec[c])
                d[key]['max'][k] = max(Vec[c])
                d[key]['min'][k] = min(Vec[c])
                d[key]['sma'][k] = sma(Vec[0], Vec[1], Vec[2])
                d[key]['energy'][k] = energy(Vec[c])
                d[key]['iqr'][k] = stats.iqr(Vec[c])
        else:
            Mag = []
            if key == 'tBodyAccMag' or key == 'tGravityAccMag':
                Mag = [ Acc[0][row], Acc[1][row], Acc[2][row] ]
            elif key == 'tBodyGyroMag':
                Mag = [ Gyro[0][row], Gyro[1][row], Gyro[2][row] ]
            elif key == 'tBodyAccJerkMag':
                Mag = [ dervAcc[0][row], dervAcc[1][row], dervAcc[2][row] ]
            elif key == 'tBodyGyroJerkMag':
                Mag = [ dervGyro[0][row], dervGyro[1][row], dervGyro[2][row] ]
            d[key]['mean'] = np.mean(Mag)
            d[key]['std'] = np.std(Mag)
            d[key]['med'] = np.median(Mag)
            d[key]['max'] = max(Mag)
            d[key]['min'] = min(Mag)
            d[key]['sma'] = sma([Mag[0]], [Mag[1]], [Mag[2]])
            d[key]['energy'] = energy(Mag)
            d[key]['iqr'] = stats.iqr(Mag)
        #endelse
    #endfor
    return d
##########################################

def makeFeatVec(Acc, Gyro, alphas, betas, row):
    #1st list of Acc/Gyro = X, 2nd list = Y, 3rd = Z
    feat_vec = []
    mask = [1, -1]
    dervAcc = [ list(ndimage.convolve(Acc[0], mask, mode='constant')),
                    list(ndimage.convolve(Acc[1], mask, mode='constant')),
                    list(ndimage.convolve(Acc[2], mask, mode='constant'))
                    ]
    dervGyro = [ list(ndimage.convolve(Gyro[0], mask, mode='constant')),
                     list(ndimage.convolve(Gyro[1], mask, mode='constant')),
                     list(ndimage.convolve(Gyro[2], mask, mode='constant'))
                     ]
    features = computeFeatures(alphas, betas, Acc, Gyro, dervAcc, dervGyro, row)

    for key in features.keys():
        for k1 in features[key].keys():
            print(key, ": ",  k1, features[key][k1])
    
    for a in alphas:
        A = getA(a, Acc, Gyro, dervAcc, dervGyro, row)
        if a.rfind('Mag') == -1:
            Vec = []
            if a == 'tBodyAcc' or a == 'tGravityAcc':
                Vec = Acc
            elif a == 'tBodyGyro':
                Vec = Gyro
            elif a == 'tBodyAccJerk':
                Vec = dervAcc
            elif a == 'tBodyGyroJerk':
                Vec = dervGyro
            for b in betas:
                for c, k in enumerate(['X', 'Y', 'Z']):
                    feat_vec.append(round(A[c] - features[a][b][k] - Vec[c][row], 4))
        else:
            for b in betas:
                feat_vec.append(round(A[0] - features[a][b], 4))
    return feat_vec

##################################
def computeFeatures2(Acc, Gyro, dervAcc, dervGyro, alphas, betas, row):
    d = createDictionary(alphas, betas)    
    for key in d.keys():
        if key.rfind('Mag') == -1:
            Vec = []
            if key == 'fBodyAcc':
                Vec = Acc
            elif key == 'fBodyGyro':
                Vec = Gyro
            elif key == 'fBodyAccJerk':
                Vec = dervAcc  
            for c, k in enumerate(['X', 'Y', 'Z']):
                d[key]['mean'][k] = np.mean(Vec[c])
                d[key]['std'][k] = np.std(Vec[c])
                d[key]['med'][k] = np.median(Vec[c])
                d[key]['max'][k] = max(Vec[c])
                d[key]['min'][k] = min(Vec[c])
                d[key]['sma'][k] = sma(Vec[0], Vec[1], Vec[2])
                d[key]['energy'][k] = energy(Vec[c])
                d[key]['iqr'][k] = stats.iqr(Vec[c])
                d[key]['maxInds'][k] = Vec[c].index(max(Vec[c]))
                d[key]['meanFreq'][k] = 0
                d[key]['skewness'][k] = stats.skew(Vec[c])
                d[key]['kurtosis'][k] = stats.kurtosis(Vec[c])
        else:
            Mag = []
            if key == 'fBodyAccMag':
                Mag = [ Acc[0][row], Acc[1][row], Acc[2][row] ]
            elif key == 'fBodyGyroMag':
                Mag = [ Gyro[0][row], Gyro[1][row], Gyro[2][row] ]
            elif key == 'fBodyAccJerkMag':
                Mag = [ dervAcc[0][row], dervAcc[1][row], dervAcc[2][row] ]
            elif key == 'fBodyGyroJerkMag':
                Mag = [ dervGyro[0][row], dervGyro[1][row], dervGyro[2][row] ]
            d[key]['mean'] = np.mean(Mag)
            d[key]['std'] = np.std(Mag)
            d[key]['med'] = np.median(Mag)
            d[key]['max'] = max(Mag)
            d[key]['min'] = min(Mag)
            d[key]['sma'] = sma([Mag[0]], [Mag[1]], [Mag[2]])
            d[key]['energy'] = energy(Mag)
            d[key]['iqr'] = stats.iqr(Mag)
            d[key]['maxInds'] = Mag.index(max(Mag))
            d[key]['meanFreq'] = 0
            d[key]['skewness'] = stats.skew(Mag)
            d[key]['kurtosis'] = stats.kurtosis(Mag)
        #endelse
    #endfor
    return d

##################################
def makeFeatVec2(Acc, Gyro, alphas, betas, feat_vec, row):
    mask = [1, -1]
    dervAcc = [ list(ndimage.convolve(Acc[0], mask, mode='constant')),
                    list(ndimage.convolve(Acc[1], mask, mode='constant')),
                    list(ndimage.convolve(Acc[2], mask, mode='constant'))
                    ]
    dervGyro = [ list(ndimage.convolve(Gyro[0], mask, mode='constant')),
                     list(ndimage.convolve(Gyro[1], mask, mode='constant')),
                     list(ndimage.convolve(Gyro[2], mask, mode='constant'))
                     ]
    features = computeFeatures2(Acc, Gyro, dervAcc, dervGyro, alphas, betas, row)
    
    for a in alphas:
        A = getA(a, Acc, Gyro, dervAcc, dervGyro, row)
        if a.rfind('Mag') == -1:
            Vec = []
            if a == 'fBodyAcc':
                Vec = Acc
            elif a == 'fBodyGyro':
                Vec = Gyro
            elif a == 'fBodyAccJerk':
                Vec = dervAcc
            for b in betas:
                for c, k in enumerate(['X', 'Y', 'Z']):
                    feat_vec.append(round(A[c] - features[a][b][k] - Vec[c][row], 4))
        else:
            for b in betas:
                feat_vec.append(round(A[0] - features[a][b], 4))
    return feat_vec
##################################
##############MAIN################

Acc, Gyro = readData('traindata.txt')

alphas = ['tBodyAcc', 'tGravityAcc', 'tBodyAccJerk', 'tBodyGyro', 'tBodyGyroJerk',
              'tBodyAccMag', 'tGravityAccMag', 'tBodyAccJerkMag', 'tBodyGyroMag', 'tBodyGyroJerkMag']  
betas = ['mean', 'std', 'med', 'max', 'min', 'sma', 'energy', 'iqr']

alphas2 = ['fBodyAcc', 'fBodyAccJerk', 'fBodyGyro', 'fBodyAccMag', 'fBodyAccJerkMag', 'fBodyGyroMag', 'fBodyGyroJerkMag']
betas2 = ['mean', 'std', 'med', 'max', 'min', 'sma', 'energy', 'iqr', 'maxInds', 'meanFreq', 'skewness', 'kurtosis']

outfilename = "featureVectors.txt"
outfile = open(outfilename, "w+")


for row in range(0, len(Acc[0])):
    feat_vec = makeFeatVec(Acc, Gyro, alphas, betas, row)       #time domain
    feat_vec = makeFeatVec2(Acc, Gyro, alphas2, betas2, feat_vec, row) #frequency domain
    writeToFile(feat_vec, outfile)
    outfile.write('\n')

outfile.close()
################END####################
















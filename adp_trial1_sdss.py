from __future__ import division
from math import *
import numpy as np
import astroML
from astroML.datasets import fetch_sdss_specgals

np.set_printoptions(threshold = 'nan')

#file = open("GalaxyZoo1_DR_table2.txt")
data = np.loadtxt("GalaxyZoo1_DR_table2.txt", delimiter = ",", skiprows = 1, usecols = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
labels = np.loadtxt("GalaxyZoo1_DR_table2.txt", delimiter = ",", skiprows = 1, usecols = (13, 14, 15)) #only gets binary label cols (spiral, elliptical, uncertain)
#top line with names of cols skipped by loadtxt:
#OBJID,RA,DEC,NVOTE,P_EL,P_CW,P_ACW,P_EDGE,P_DK,P_MG,P_CS,P_EL_DEBIASED,P_CS_DEBIASED,SPIRAL,ELLIPTICAL,UNCERTAIN

counter = 0
for line in data:
    counter += 1

print "Number of galaxies studied:", counter
#print data.dtype.name['elliptical']


#Extracting sdss data to compare galazyzoo data against
data2 = fetch_sdss_specgals()
print data2.dtype.names
print "Number of attributes for each galaxy:", len(data2.dtype.names)
print "Number of galaxies included:", len(data2)
#galaxies selected here all have quasars - 2011 Z.Ivezic Quasar selection based on photometric variability

print labels

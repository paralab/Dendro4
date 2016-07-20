# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# @author: Milinda Fernando
# School of Computing, University of Utah.
# generate all the slurm jobs for the sc16 poster, energy measurements,

import argparse
from subprocess import call
import os
import numpy as np
import csv
import json
from numpy import genfromtxt

if __name__ == "__main__":
    
 parser = argparse.ArgumentParser(prog='slurm_pbs')
 parser.add_argument('-i','--input', help='file prefix that you need to merge')
 parser.add_argument('-n','--n',help='number of flies that you need to merge')
 args=parser.parse_args()
 
 csvname=args.input+'.csv' #'sendCommMap_M_tol_0.010000_npes_32_pts_1000_ps_32mat.csv'
 #csvfile  = open('sendCommMap_H_tol_0.001000_npes_4096_pts_100000_ps_4096mat.csv','r')
 
 
 my_data = genfromtxt(csvname, delimiter=',')
 



# <codecell>

 i=0
 gf=8
 n=int(args.n)
 #jsonfile = open('sendCommMap_M_tol_0.010000_npes_32_pts_1000_ps_32mat.json', 'w')
 jsonfile = open(args.input+'.json', 'w')
 jsonfile.write('[\n')
 for row in my_data:
        com_map=np.where(row!=0)[0]
        com_map_str = ','.join(['\"P.%02d.%02d\"' % ((num/gf), (num)) for num in com_map])
        if(i<(n-1)):
            js_str='{\"name\":\"P.'+str(i/gf).zfill(2)+'.'+str(i).zfill(2)+'\",\"size\":'+str(np.count_nonzero(row))+',\"imports\":['+com_map_str+']},\n'
        else:
            js_str='{\"name\":\"P.'+str(i/gf).zfill(2)+'.'+str(i).zfill(2)+'\",\"size\":'+str(np.count_nonzero(row))+',\"imports\":['+com_map_str+']}\n'
        #print com_map_str
        #print js_str
        jsonfile.write(js_str)
        i=i+1
        #break
 jsonfile.write(']')
 jsonfile.close()
        

# <codecell>



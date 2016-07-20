# @author: Milinda Fernando
# School of Computing, University of Utah.
# generate all the slurm jobs for the sc16 poster, energy measurements,

import argparse
from subprocess import call
import os

if __name__ == "__main__":
 parser = argparse.ArgumentParser(prog='slurm_pbs')
 parser.add_argument('-p','--prefix', help='file prefix that you need to merge')
 parser.add_argument('-s','--suffix',help='suffix of the file')
 parser.add_argument('-n','--n',help='number of flies that you need to merge')
 args=parser.parse_args()

 tol_list=['0.000010','0.000100','0.001000','0.010000','0.100000','0.200000','0.300000','0.400000','0.500000']	 
 #sendCommMap_M_tol_0.010000_npes_4096_pts_100000_ps_4096mat.csv

 for tol in tol_list:
	inFName=args.prefix+tol+args.suffix+'_'+args.n+'mat'+'.csv'
	outFName=args.prefix+tol+args.suffix+'_'+args.n+'mat_comma'+'.csv'
 	fin=open(inFName,'r')
 	fout=open(outFName,'w')
 	for line in fin:
		line=line.strip()
		line=line.replace('\t',',')	
		fout.write(line+'\n')
 	 fin.close()
 	fout.close()
 print 'OK'	

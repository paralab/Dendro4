# @author: Milinda Fernando
# School of Computing, University of Utah.
# generate all the slurm jobs for the sc16 poster, energy measurements,

import argparse
from subprocess import call
import os

if __name__ == "__main__":
 parser = argparse.ArgumentParser(prog='slurm_pbs')
 parser.add_argument('-p','--prefix', help='file prefix that you need to merge')
 parser.add_argument('-n','--n',help='number of flies that you need to merge')
 args=parser.parse_args()
 
 outFName=args.prefix+'_'+args.n+'mat'+'.csv'
 fout=open(outFName,'w')
 for i in range(0,int(args.n)):
	inFName=args.prefix+str(i)+'_'+args.n+'.csv'
	f=open(inFName)
	line=f.read()
	line=line.strip()
	line=line.replace('\t',',')
	fout.write(line+'\n')
	f.close()
 fout.close()
 print 'file merge .. OK'	

 

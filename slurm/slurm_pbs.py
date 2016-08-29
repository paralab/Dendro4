# @author: Milinda Fernando
# School of Computing, University of Utah.
# generate all the slurm jobs for the sc16 poster, energy measurements,

import argparse


if __name__ == "__main__":
 parser = argparse.ArgumentParser(prog='slurm_pbs')
 parser.add_argument('-n','--npes', help=' number of mpi tasks')
 parser.add_argument('-N','--N',help='number of cpu cores per single mpi task')
 parser.add_argument('-g','--grainSz',help='grain size of the problem')
 parser.add_argument('-i','--Iterations',help='number of iterations for the matvec operation')
 #parser.add_argument('--tolBegin',help='tol begin value')
 #parser.add_argument('--tolEnd', help='tol end value')
 #parser.add_argument('--tolStep',help='tol step')
 parser.add_argument('--SFC',help='SFC method H or M')

 args=parser.parse_args()
 TolList=[1e-5,1e-4,1e-3,1e-2,1e-1,2e-1,3e-1]

 for tol in TolList:
     fileName='sc16_fpart_'+args.SFC+'_'+str(tol)+'.pbs'
     pbs_file=open(fileName,'w')
     pbs_file.write('#!/bin/bash\n')
     pbs_file.write('#SBATCH --ntasks='+args.npes+'\n')
     pbs_file.write('#SBATCH --cpus-per-task='+args.N+'\n')
     pbs_file.write('#SBATCH -o /exp-share/dendro/build/sc16-poster-final-jobs/%J.out\n')
     pbs_file.write('#SBATCH -e /exp-share/dendro/build/sc16-poster-final-jobs/%J.err\n')
     pbs_file.write('#SBATCH --time=24:00:00\n')
     pbs_file.write('#SBATCH --account=perf\n')
     pbs_file.write('n='+args.N+'\n')
     pbs_file.write('inputFile=ip\n')
     pbs_file.write('numPts='+args.grainSz+'\n')
     pbs_file.write('dim=3\n')
     pbs_file.write('maxDepth=30\n')
     pbs_file.write('solvU=0\n')
     pbs_file.write('writeB=0\n')
     pbs_file.write('k=1\n')
     pbs_file.write('inCorner=1\n')
     pbs_file.write('numLoops='+args.Iterations+'\n')
     pbs_file.write('compress=0\n')
     pbs_file.write('export OMP_NUM_THREADS=1\n')
     pbs_file.write('cd /exp-share/dendro/build\n')
     pbs_file.write('tol=0.000001\n')
     if args.SFC=='H':
         pbs_file.write('echo \'Executing Hilbert\'\n')
         pbs_file.write('mpirun --allow-run-as-root -np $n ./tstTreeSortMatVec_h $inputFile $numPts $dim $maxDepth $tol 1 $solvU $writeB $k $inCorner $numLoops $compress\n')
     elif args.SFC=='M':
         pbs_file.write('echo \'Executing Morton\'\n')
         pbs_file.write('mpirun --allow-run-as-root -np $n ./tstTreeSortMatVec_m $inputFile $numPts $dim $maxDepth $tol 1 $solvU $writeB $k $inCorner $numLoops $compress\n')

     pbs_file.close()


















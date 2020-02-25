#!/usr/bin/env python

import os

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = "%s/jobs" %os.getcwd()
# Make top level directories
mkdir_p(job_directory)

script_path= "$HOME/projects/seasonal_forecasting/code/scripts/"
fields = ["swvl1", "msl", "t2m"]
regions = ["EU", "NA-EU", "EU"]
preprocesses = ["anomalies", "anomalies", "anomalies"]
lrs = [1e-2, 1e-9, 5e-5]
n_min,n_max = 1, 5

for field, region, preprocess, lr in zip(fields, regions, preprocesses, lrs):
    for n in range(n_min,n_max+1):
        jn = field + '_' + region + '_' + preprocess + '_n' + str(n)
        job_file = os.path.join(job_directory,"%s.job" %jn)
        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/tcsh\n")
            fh.writelines("#SBATCH --job-name=%s.job\n" % jn)
            fh.writelines("#SBATCH --partition=pCluster")
            fh.writelines("#SBATCH --N=1")
            fh.writelines("#SBATCH --n=40")
            fh.writelines("#SBATCH --t=30\n")
            fh.writelines("#SBATCH --output=jobs/%s.out\n" % jn)
            fh.writelines("#SBATCH --error=jobs/%s.err\n" % jn)
            fh.writelines("#SBATCH --mail-type=END\n")
            fh.writelines("#SBATCH --mail-user=nonnenma@hzg.de\n")
            fh.writelines("#SBATCH --account=nonnenma\n")    
            fh.writelines(f"python {script_path}script_pCluster_run_single.py {n} {field} {region} {preprocess} {lr}\n")

        os.system("sbatch %s" %job_file)
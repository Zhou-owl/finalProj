
from __future__ import division, absolute_import, unicode_literals, print_function

import argparse
import sys
import os
import shutil

sys.path.append(os.path.abspath("../"))

from DisneyDisp import Disney




################################################################################
#                                                                              #
#                       Can be used as a command line tool                     #
#                                                                              #
################################################################################

parser = argparse.ArgumentParser(description='Calulate disparity fom a given lightfield.')

parser.add_argument('lightfield', help='The filename including the directory of the lightfield. A lightfield must be a .hdf5 file with one dataset in the root directory. The dataset must contain n images, i.e. we have dimensions [s,v,u].')

parser.add_argument('--hdf5_dataset', help='The dataset name inside the .hdf5 File.', default='lightfield')
parser.add_argument('--working_dir', help='The directory used to store intermediate data.', default='tmp/')
parser.add_argument('--result_dir', help='The output directory.', default='results/')
parser.add_argument('--s_hat', help='If given, only the disparity of image from this s  dimension will be computed.', type=int, default=50)
parser.add_argument('--r_start', help='The resolution to start with. Can only be used in DEBUG mode.',nargs=2, type=int, default=None)
parser.add_argument('--min_disp', help='The minimal disparity to sample for.', type=float, default=0)
parser.add_argument('--max_disp', help='The maximal disparity to sample for.', type=float, default=20)
parser.add_argument('--stepsize', help='The disparity step size during sampling .', type=float, default=0.1)
parser.add_argument('--Ce_t', help='The threshold for the edge confidence measurement.', type=float, default=0.02)
parser.add_argument('--Cd_t', help='The threshold for the depth confidence measurement.', type=float, default=0.1)
parser.add_argument('--S_t', help='The threshold for the bilateral median filter measurement.',type=float, default=0.1)
parser.add_argument('--n_jobs', help='The number of threads to use.', type=int, default=-1)
parser.add_argument('-NOISEFREE', help='Disable radiance update for lightfield without noise', action='store_true')
parser.add_argument('-DEBUG', help='Enable plotting of intermediate results',action='store_true')


if __name__ == "__main__":
    terminal = 0
    if terminal:
        args = parser.parse_args()

        disney = Disney(args.lightfield, args.hdf5_dataset, args.result_dir,
                        working_dir=args.working_dir, n_cpus=args.n_jobs,
                        r_start=args.r_start, s_hat=args.s_hat, DEBUG=args.DEBUG)

        disney.calculateDisp(min_disp=args.min_disp, max_disp=args.max_disp,
                            stepsize=args.stepsize, Ce_t=args.Ce_t, Cd_t=args.Cd_t,
                            S_t=args.S_t, NOISEFREE=args.NOISEFREE)
        disney.calculateMap()
    else:
        disney = Disney("./bicycle.hdf5", "bicycle", "results_full_bicycle/",
                        working_dir='tmp2/', n_cpus=-1,
                        r_start=None, s_hat=50, DEBUG=True)

        disney.calculateDisp(min_disp=0, max_disp=20,
                            stepsize=0.5, Ce_t=2, Cd_t=0.1,
                            S_t=2, NOISEFREE=True)
        disney.calculateMap()   
    shutil.rmtree("/home/sc/Desktop/yongqi/depth/Depth-Estimation-Light-Field/DisneyDispPy-master/tmp2")


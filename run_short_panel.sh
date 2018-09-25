#!/bin/bash


# NOTE: For run-time performance measurements, comment out the parallel
#   operator '&' so that each run gets the same resources.


NUM_RUNS=10
MAX_STEP=15000
LOG_STEP=1000

################################################################################
# CE GAN
if [ "$1" == 'ce_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/ce_iw_dim2_run${i}; python gan_ce_iw.py --tag='iw_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 --estimator='iw' &
    rm -rf results/ce_iw_dim4_run${i}; python gan_ce_iw.py --tag='iw_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 --estimator='iw'  &
    rm -rf results/ce_iw_dim10_run${i}; python gan_ce_iw.py --tag='iw_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 --estimator='iw' &
  done
fi

if [ "$1" == 'ce_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/ce_miw_dim2_run${i}; python gan_ce_miw.py --tag='miw_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2  &
    rm -rf results/ce_miw_dim4_run${i}; python gan_ce_miw.py --tag='miw_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4  &
    rm -rf results/ce_miw_dim10_run${i}; python gan_ce_miw.py --tag='miw_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10  &
  done
fi

if [ "$1" == 'ce_sniw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/ce_sniw_dim2_run${i}; python gan_ce_iw.py --tag='sniw_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 --estimator='sniw' &
    rm -rf results/ce_sniw_dim4_run${i}; python gan_ce_iw.py --tag='sniw_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 --estimator='sniw' &
    rm -rf results/ce_sniw_dim10_run${i}; python gan_ce_iw.py --tag='sniw_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 --estimator='sniw' &
  done
fi

# CONDITIONAL
if [ "$1" == 'ce_conditional' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/ce_conditional_dim2_run${i}; python gan_ce_conditional.py --tag='conditional_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 &
    rm -rf results/ce_conditional_dim4_run${i}; python gan_ce_conditional.py --tag='conditional_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 &
    rm -rf results/ce_conditional_dim10_run${i}; python gan_ce_conditional.py --tag='conditional_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 &
  done
fi

# UPSAMPLE
if [ "$1" == 'ce_upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/ce_upsample_dim2_run${i}; python gan_ce_upsample.py --tag='upsample_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 &
    rm -rf results/ce_upsample_dim4_run${i}; python gan_ce_upsample.py --tag='upsample_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 &
    rm -rf results/ce_upsample_dim10_run${i}; python gan_ce_upsample.py --tag='upsample_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 &
  done
fi

################################################################################
# MMD GAN
if [ "$1" == 'mmd_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/mmd_iw_dim2_run${i}; python gan_mmd_iw.py --tag='iw_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 --estimator='iw' &
    rm -rf results/mmd_iw_dim4_run${i}; python gan_mmd_iw.py --tag='iw_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 --estimator='iw' &
    rm -rf results/mmd_iw_dim10_run${i}; python gan_mmd_iw.py --tag='iw_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 --estimator='iw' &
  done
fi

if [ "$1" == 'mmd_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/mmd_miw_dim2_run${i}; python gan_mmd_miw.py --tag='miw_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 &
    rm -rf results/mmd_miw_dim4_run${i}; python gan_mmd_miw.py --tag='miw_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 &
    rm -rf results/mmd_miw_dim10_run${i}; python gan_mmd_miw.py --tag='miw_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 &
  done
fi

if [ "$1" == 'mmd_sniw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/mmd_sniw_dim2_run${i}; python gan_mmd_iw.py --tag='sniw_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 --estimator='sniw' &
    rm -rf results/mmd_sniw_dim4_run${i}; python gan_mmd_iw.py --tag='sniw_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 --estimator='sniw' &
    rm -rf results/mmd_sniw_dim10_run${i}; python gan_mmd_iw.py --tag='sniw_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 --estimator='sniw' &
  done
fi

# UPSAMPLE
if [ "$1" == 'mmd_upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/mmd_upsample_dim2_run${i}; python gan_mmd_upsample.py --tag='upsample_dim2_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=2 &
    rm -rf results/mmd_upsample_dim4_run${i}; python gan_mmd_upsample.py --tag='upsample_dim4_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=4 &
    rm -rf results/mmd_upsample_dim10_run${i}; python gan_mmd_upsample.py --tag='upsample_dim10_run'${i} --max_step=$MAX_STEP --log_step=$LOG_STEP --data_dim=10 &
  done
fi

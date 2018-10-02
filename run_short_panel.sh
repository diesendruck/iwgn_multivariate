#!/bin/bash


# This script defines specific models to run. It is called by the main script
# called run_panel.sh.
#
# NOTE: For run-time performance measurements, comment out the parallel
#   operator '#&' so that each run gets the same resources.


NUM_RUNS=10
MAX_STEP=20001
LOG_STEP=1000
BATCH_SIZE=64
LEARNING_RATE=1e-3
DO_DIM2=true
DO_DIM4=true
DO_DIM10=true


################################################################################
# CE GAN
if [ "$1" == 'ce_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_iw_dim2_run${i}; python gan_ce_iw.py \
        --tag='iw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --estimator='iw'   &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_iw_dim4_run${i}; python gan_ce_iw.py \
        --tag='iw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --estimator='iw'   &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_iw_dim10_run${i}; python gan_ce_iw.py \
        --tag='iw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --estimator='iw'   &
    fi
  done
fi

if [ "$1" == 'ce_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_miw_dim2_run${i}; python gan_ce_miw.py \
        --tag='miw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2   &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_miw_dim4_run${i}; python gan_ce_miw.py \
        --tag='miw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4   &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_miw_dim10_run${i}; python gan_ce_miw.py \
        --tag='miw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10   &
    fi
  done
fi

if [ "$1" == 'ce_sniw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_sniw_dim2_run${i}; python gan_ce_iw.py \
        --tag='sniw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --estimator='sniw'   &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_sniw_dim4_run${i}; python gan_ce_iw.py \
        --tag='sniw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --estimator='sniw'   &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_sniw_dim10_run${i}; python gan_ce_iw.py \
        --tag='sniw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --estimator='sniw'   &
    fi
  done
fi

# CONDITIONAL
if [ "$1" == 'ce_conditional' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_conditional_dim2_run${i}; python gan_ce_conditional.py \
        --tag='conditional_dim2_run'${i} --max_step=$MAX_STEP --data_dim=2 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP   &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_conditional_dim4_run${i}; python gan_ce_conditional.py \
        --tag='conditional_dim4_run'${i} --max_step=$MAX_STEP --data_dim=4 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP   &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_conditional_dim10_run${i}; python gan_ce_conditional.py \
        --tag='conditional_dim10_run'${i} --max_step=$MAX_STEP --data_dim=10 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP   &
    fi
  done
fi

# UPSAMPLE
if [ "$1" == 'ce_upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_upsample_dim2_run${i}; python gan_ce_upsample.py \
        --tag='upsample_dim2_run'${i} --max_step=$MAX_STEP --data_dim=2\
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --sampling='random'   &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_upsample_dim4_run${i}; python gan_ce_upsample.py \
        --tag='upsample_dim4_run'${i} --max_step=$MAX_STEP --data_dim=2 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --sampling='random'   &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_upsample_dim10_run${i}; python gan_ce_upsample.py \
        --tag='upsample_dim10_run'${i} --max_step=$MAX_STEP --data_dim=10\
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --sampling='random'   &
    fi
  done
fi
#if [ "$1" == 'ce_upsample_rej' ]; then
#  for i in $( seq 1 $NUM_RUNS ); do
#    rm -rf results/ce_upsample_rej_dim2_run${i}; python gan_ce_upsample.py \
#      --tag='upsample_rej_dim2_run'${i} --max_step=$MAX_STEP --data_dim=2\
#      --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
#      --log_step=$LOG_STEP --sampling='rejection'   #&
#    #rm -rf results/ce_upsample_dim4_run${i}; python gan_ce_upsample.py \
#    #  --tag='upsample_dim4_run'${i} --max_step=$MAX_STEP --data_dim=2 \
#    #  --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
#    #  --log_step=$LOG_STEP --sampling='rejection'   #&
#    #rm -rf results/ce_upsample_dim10_run${i}; python gan_ce_upsample.py \
#    #  --tag='upsample_dim10_run'${i} --max_step=$MAX_STEP --data_dim=10\
#    #  --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
#    #  --log_step=$LOG_STEP --sampling='rejection'   #&
#  done
#fi


################################################################################
# MMD GAN
if [ "$1" == 'mmd_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_iw_dim2_run${i}; python gan_mmd_iw.py --data_dim=2 \
        --tag='iw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --estimator='iw'   #&
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_iw_dim4_run${i}; python gan_mmd_iw.py --data_dim=4 \
        --tag='iw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --estimator='iw'   #&
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_iw_dim10_run${i}; python gan_mmd_iw.py --data_dim=10 \
        --tag='iw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --estimator='iw'   #&
    fi
  done
fi

if [ "$1" == 'mmd_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_miw_dim2_run${i}; python gan_mmd_miw.py \
        --tag='miw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2   #&
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_miw_dim4_run${i}; python gan_mmd_miw.py \
        --tag='miw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4   #&
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_miw_dim10_run${i}; python gan_mmd_miw.py \
        --tag='miw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10   #&
    fi
  done
fi

if [ "$1" == 'mmd_sniw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_sniw_dim2_run${i}; python gan_mmd_iw.py \
        --tag='sniw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --estimator='sniw'   #&
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_sniw_dim4_run${i}; python gan_mmd_iw.py \
        --tag='sniw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --estimator='sniw'   #&
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_sniw_dim10_run${i}; python gan_mmd_iw.py \
        --tag='sniw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --estimator='sniw'   #&
    fi
  done
fi

# UPSAMPLE
if [ "$1" == 'mmd_upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_upsample_dim2_run${i}; python gan_mmd_upsample.py \
        --tag='upsample_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2   #&
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_upsample_dim4_run${i}; python gan_mmd_upsample.py \
        --tag='upsample_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4   #&
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_upsample_dim10_run${i}; python gan_mmd_upsample.py \
        --tag='upsample_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10   #&
    fi
  done
fi

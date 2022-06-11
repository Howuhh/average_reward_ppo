#!/usr/bin/env bash
export PYTHONPATH=.

#seeds=(1 10 20 30 50 60 70 80 90 100)
seeds=(1 10 20)

for seed in "${seeds[@]}"
do
  python scripts/train.py --project=PPO --entity=howuhh --group=test2_cheetah_multiseed --name=apo \
    --env_name=HalfCheetah-v3 \
    --num_epochs=15 \
    --clip_grad=10 \
    --value_constraint=1.0 \
    --value_loss_coef=0.5 \
    --gae_lambda=0.9 \
    --tau=0.1 \
    --clip_eps=0.3 \
    --train_seed=$seed
done

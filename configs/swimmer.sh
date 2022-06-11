#!/usr/bin/env bash
export PYTHONPATH=.

seeds=(1 10 20 30 50 60 70 80 90 100)

for seed in "${seeds[@]}"
do
  python scripts/train.py --project=PPO --entity=howuhh --group=best_swimmer_multiseed --name=apo \
    --env_name=Swimmer-v3 \
    --gae_lambda=0.99 \
    --clip_grad=10.0 \
    --value_constraint=1.0 \
    --value_loss_coef=1.0 \
    --train_seed=$seed
done


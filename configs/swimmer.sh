export PYTHONPATH=..

seeds=(1 2 3 4 5 6 7 8 9 10)

for seed in "${seeds[@]}"
do
  scripts/train.py --project=PPO --entity=howuhh --group=best_swimmer_multiseed --name=apo
  --env_name=Swimmer-v3 \
  --gae_lambda=0.99 \
  --clip_grad=10.0 \
  --value_constraint=1.0 \
  --value_loss_coef=1.0 \
  --seed=$seed
done


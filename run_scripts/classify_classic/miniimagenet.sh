
source activate pnl

d=miniimagenet
p=10
lr=1e-1
for o  in adam sgd; do
for a in  1 0; do
for lr in 1e-1; do
for b in 128; do
for s in 0; do

cmd="classify_classic.py \
--dataroot data \
--ckpt_every 50 \
--epochs  200 \
--dataset $d \
--optim $o \
--seed $s \
--output_dir results/lcbo/$d/$d-$o-$a-$lr-$b-$s \
--lr $lr \
--pdim $p \
--n_eval_batches 10 \
--batch-size $b  \
--test-batch-size 1000 \
--data_augmentation $a "

python $cmd


done
done
done
done
done


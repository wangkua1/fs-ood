source activate pnl

d=cifar-fs-train
p=10
lr=1e-1
r=1
db=0
for o  in sgd; do
for a in  1 ; do
for lr in 1e-1; do
for b in 128; do
for s in 0; do
cmd="classify_classic.py \
--ckpt_every 1 \
--epochs  200 \
--dataset $d \
--optim $o \
--seed $s \
--output_dir $ROOT1/fs-ood-submitted/$d-ensemble-${db}/$d-$o-$a-$lr-$b-$s-$r \
--lr $lr \
--dataroot $ROOT1/data/ \
--pdim $p \
--n_eval_batches 10 \
--batch-size $b  \
--resume ${r} \
--db ${db} \
--test-batch-size 1000 \
--data_augmentation $a"



python $cmd


done
done
done
done
done


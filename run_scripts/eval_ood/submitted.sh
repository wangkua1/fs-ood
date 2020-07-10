

d=cifar-fs-train
p=10
lr=1e-1
o=sgd
a=1 
lr=1e-1
b=128
s=0


model_dir=$ROOT1/fs-ood-submitted/$d-ensemble/$d-$o-$a-$lr-$b-$s
model_path=${model_dir}/ckpt_best.pt


db=0
d=cifar-fs

for ood in MPP DM-all native-spp native-ed deep-ed-iso oec; do
for e in  meta-test; do

cmd="eval_ood.py \
--classifier_path ${model_path} \
--fsmodel_name baseline-pn \
--fsmodel_path - \
--glow_dir - \
--oec_path - \
--output_dir ${ROOT1}/fs-ood-submitted/eval_ood/$d \
--dataset $d \
--dataroot data \
--episodic_ood_eval 1 \
--episodic_in_distr $e \
--ood_methods $ood \
--n_episodes 100 \
--db $db
" 

python $cmd

done
done








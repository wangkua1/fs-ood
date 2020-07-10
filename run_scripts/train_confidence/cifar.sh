
d=cifar-fs-train
p=10
lr=1e-1
o=sgd
a=1 
lr=1e-1
b=128
s=0
r=1
db=0

model_dir=results/$d-ensemble-${db}/$d-$o-$a-$lr-$b-$s-$r 
model_path=${model_dir}/ckpt_best.pt

d=cifar-fs
for tq in 15; do
for lr in 1e-3; do
for arch in  "500,500"; do
for n in 'bn'; do
for m in deep-oec; do
for way in 2 3 4 5 6 7 8; do
for shot in 1 2 5 10 20 50; do

name=${m}-${way}-${shot}-${arch}-${lr}
oec_dir=results/train_confidence-ways-shots-grid/${name}
oec_path=${oec_dir}/${name}_conf_best.pt
cmd="train_confidence.py \
	--output_dir $oec_dir \
    --model.model_path $model_path \
    --train_iter 5000 \
    --data.test_episodes 10 \
    --data.test_query ${tq} \
    --data.test_way ${way} \
    --data.test_shot ${shot} \
    --lr ${lr} \
    --residual 1 \
    --arch ${arch} \
    --exp_name ${name} \
    --confidence_method $m \
    --oec_embed_in_eval 0 \
    --oec_norm_type $n \
    --dataroot data \
    --dataset $d "



python $cmd


done
done
done
done
done
done
done

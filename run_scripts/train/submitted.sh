
mc=protonet
# mc=maml

for d in cifar100 miniimagenet; do
for enc in conv4 resnet10; do
for lr in 1e-3; do
for a in 0 1; do
name=${enc}-${a}-${lr}
cmd="train.py \
	--data.dataset $d \
	--model.encoder $enc \
	--model.class ${mc} \
	--train.decay_every 1000000 \
	--data.test_way 5 \
	--data.test_shot 5 \
	--data.way 5 \
	--data.shot 5 \
	--model.f_acq spp \
	--log.exp_dir ${ROOT1}/lcbo/train/$d-protonet/${name} \
	--dataroot 'data' \
	--train.learning_rate $lr \
	--train.optim_method Adam \
	--train.weight_decay 0 \
	--data_augmentation $a 
"


python $cmd


done
done
done
done
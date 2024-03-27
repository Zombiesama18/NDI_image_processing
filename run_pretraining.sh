seed=19981303
epochs=1000
save_steps=100
model_name="resnet50_imagenet21k"
input_size=200
batch_size=128
learning_rate=5e-3

dataset_dir="../datasets/NDI_images/Integreted"
output_dir="./checkpoints/"
log_dir="./logs/"


torchrun --nnodes 1 --nproc_per_node 1 pretraining.py \
    --dataset_dir $dataset_dir \
    --output_dir $output_dir \
    --log_dir $log_dir \
    --model $model_name \
    --input_size $input_size \
    --seed $seed \
    --batch_size $batch_size \
    --lr $learning_rate \
    --epochs $epochs \
    --save_steps $save_steps

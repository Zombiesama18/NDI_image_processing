seed=19981303
epochs=50
batch_size=16
base_lr=0.005
weight_decay=1e-4
accumulate_step=1
momentum=0.9

output_dir="./checkpoints/"
log_dir="./logs/"

torchrun --nnodes 1 --nproc_per_node 1 finetuning_on_NDI.py \
    --output_dir $output_dir \
    --seed $seed \
    --per_device_train_batch_size $batch_size \
    --learning_rate $base_lr \
    --weight_decay $weight_decay \
    --log_dir $log_dir \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps $accumulate_step \
    --momentum $momentum






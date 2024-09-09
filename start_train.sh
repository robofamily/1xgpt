accelerate launch train.py \
--train_data_dir data/train_calvin \
--val_data_dir data/validation_calvin \
--genie_config genie/configs/magvit_n32_h8_d256.json \
--per_device_train_batch_size 12 \
--per_device_eval_batch_size 4 \
--num_train_epochs 7 \
--window_size 16 \
--stride 3 \
--output_dir data/genie_model \
--lr_scheduler_type linear \
--learning_rate 3e-4 \
--num_warmup_steps 1000 \
--per_device_num_workers 12 \
--tokenizer_ckpt /public/home/muyao/1x_train/Open-MAGVIT2/imagenet_128_B.ckpt \
--depth_decoder_ckpt /public/home/muyao/1x_train/1xgpt2/data/magvit2dpt_pretrained/magvit2dpt_0.pth \
--vis_every_n_steps 10 \
--eval_every_n_steps 3000 \
--max_eval_steps 10 \
python genie/generate.py \
--val_data_dir data/validation_calvin \
--checkpoint_dir /public/home/muyao/1x_train/1xgpt/data/genie_model/step_40000 \
--output_dir data/genie_generated \
--num_prompt_frames 1 \
--example_ind 129 \
--stride 3 \
--maskgit_steps 1;

python visualize.py \
--token_dir data/genie_generated \
--tokenizer_ckpt /public/home/muyao/1x_train/Open-MAGVIT2/imagenet_128_B.ckpt \
--depth_decoder_ckpt data/magvit2dpt_pretrained/magvit2dpt_0.pth \
--offset 0 \
--generated_data \
--draw_point_cloud

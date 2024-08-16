python genie/generate.py --val_data_dir data/validation_calvin --checkpoint_dir /public/home/muyao/new_1x_train/1xgpt/data/genie_model/step_91000 --output_dir data/genie_generated --num_prompt_frames 1 --example_ind 0 --stride 1 --maskgit_steps 2;

python visualize.py --token_dir data/genie_generated --generated_data;
# python visualize.py --token_dir data/validation_calvin --max_images 100;

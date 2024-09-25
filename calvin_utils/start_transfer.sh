python transfer_lmdb_to_1x.py \
--in_dir /public/home/muyao/1x_train/calvin_lmdb_with_depth \
--out_dir ./valid_calvin \
--magvit_ckpt_path /public/home/muyao/1x_train/Open-MAGVIT2/imagenet_128_B.ckpt \
--start_ratio 0.99 \
--end_ratio 1
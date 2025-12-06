import os

os.system(
    'python train.py '
    '--data_path "C:\\Users\\konda\\Desktop\\dataset" '
    '--vit_weights "C:\\Users\\konda\\Desktop\\ThesisCode\\model_weights\\vit_base_patch16_384.npz" '
    '--device cuda '
    '--num_classes 2 '
    '--val_ratio 0.15 '
    '--test_ratio 0.5 '
    '--batch_size 16 '
    '--img_res 384 '
    '--data_info '
    '--output_test '
    '--save_dir "C:\\Users\\konda\\Desktop\\ThesisCode\\checkpoints" '
    '--epochs 20 '
    '--lr_vit 1e-5 '
    '--lr_dec 1e-4 '
    '--beta1 0.85 '
    '--beta2 0.999 '
    '--wd_vit 1e-5 '
    '--wd_dec 1e-5 '
    '--amp'
)
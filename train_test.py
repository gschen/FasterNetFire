import os
import sys
import torch
from torch import nn
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from utils.utils import *
from utils.fuse_conv_bn import fuse_conv_bn
from data.data_api import LitDataModule
from models.model_api import LitModel

def main(args):
    # Init data pipeline
    dm, _ = LitDataModule(hparams=args)

    # Init LitModel
    if args.checkpoint_path is not None:
        PATH = args.checkpoint_path
        if PATH[-5:]=='.ckpt':
            model = LitModel.load_from_checkpoint(PATH, map_location='cpu', num_classes=dm.num_classes, hparams=args)
            print('Successfully load the pl checkpoint file.')
            if args.pl_ckpt_2_torch_pth:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.model.to(device)
                torch.save(model.state_dict(), PATH[:-5]+'.pth')
                exit()
        elif PATH[-4:] == '.pth':
            model = LitModel(num_classes=dm.num_classes, hparams=args)
            missing_keys, unexpected_keys = model.model.load_state_dict(torch.load(PATH), False)
            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)
        else:
            raise TypeError
    else:
        model = LitModel(num_classes=dm.num_classes, hparams=args)

    flops, params = get_flops_params(model.model, args.image_size)

    if args.fuse_conv_bn:
        fuse_conv_bn(model.model)

    if args.measure_latency:
        dm.prepare_data()
        dm.setup(stage="test")
        for idx, (images, _) in enumerate(dm.test_dataloader()):
            model = model.model.eval()
            throughput, latency = measure_latency(images[:1, :, :, :], model, GPU=False, num_threads=1)
            if torch.cuda.is_available():
                throughput, latency = measure_latency(images, model, GPU=True)
            exit()

    # print_model(model)

    # Callbacks
    MONITOR = 'val_acc1'
    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR,
        dirpath=args.model_ckpt_dir,
        filename=args.model_name+'-{epoch}-{val_acc1:.4f}',
        save_top_k=1,
        save_last=True,
        mode='max' if 'acc' in MONITOR else 'min'
    )
    refresh_callback = TQDMProgressBar(refresh_rate=20)
    callbacks = [
        checkpoint_callback,
        refresh_callback
    ]

    # Print model info
    print(f"Model FLOPs: {flops}, Parameters: {params}")

    # Initialize a trainer
    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        max_epochs=args.epochs,
        devices=1,  # 使用单GPU
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=args.clip_grad,
        callbacks=callbacks,
        precision=args.precision,
        benchmark=args.benchmark
    )

    if bool(args.test_phase):
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, dm)
        if args.dev==0:
            # 若未产生 best ckpt（例如训练步数为 0），回退为测试当前内存权重，避免直接报错中断
            best_path = checkpoint_callback.best_model_path
            if best_path and os.path.isfile(best_path):
                trainer.test(ckpt_path="best", datamodule=dm)
            else:
                print("[warn] best checkpoint not found, fallback to current model weights for test.")
                trainer.test(model=model, datamodule=dm)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/fasternetfire.yaml')
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument('-n', "--num_workers", type=int, default=0)
    parser.add_argument('-b', "--batch_size", type=int, default=128)
    parser.add_argument('-e', "--batch_size_eva", type=int, default=1000, help='batch_size for evaluation')
    parser.add_argument('--drop_last', action='store_true',
                        help='drop last incomplete training batch (default: False)')
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument("--data_dir", type=str, default="/data/FD-dataset")
    parser.add_argument('--data_split_seed', type=int, default=42,
                        help='seed for auto split when data_dir is class-folder format')
    parser.add_argument('--data_split_train_ratio', type=float, default=0.7,
                        help='auto split ratio for train set')
    parser.add_argument('--data_split_val_ratio', type=float, default=0.2,
                        help='auto split ratio for validation set')
    parser.add_argument('--data_split_test_ratio', type=float, default=0.1,
                        help='auto split ratio for test set')
    parser.add_argument('--eval_resize_only', action='store_true',
                        help='validation/test: resize directly to image_size (no center crop)')
    parser.add_argument('--no_bowfire_flat_train', action='store_true',
                        help='BoWFire: 禁止扁平 train/（必须 train/fire 与 train/no_fire 等子目录）')
    parser.add_argument('--no_bowfire_flat_test', action='store_true',
                        help='BoWFire: 禁止扁平 dataset/img（必须含类别子目录）')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--pconv_fw_type", type=str, default='split_cat',
                        help="use 'split_cat' for training/inference and 'slicing' only for inference")
    parser.add_argument('--measure_latency', action='store_true', help='measure latency or throughput')
    parser.add_argument('--test_phase', action='store_true')
    parser.add_argument('--fuse_conv_bn', action='store_true')
    parser.add_argument('--pl_ckpt_2_torch_pth', action='store_true',
                        help='convert pl .ckpt file to torch .pth file, and then exit')
    parser.add_argument('--save_error_images', action='store_true',
                        help='test: error_inf.txt + P(pos) naming; save misclassified images (FP/FN); confusion_scatter.png')
    parser.add_argument('--error_output_dir', type=str, default='20260511error_output',
                        help='output dir for misclassified images, error_inf.txt, confusion_scatter.png')
    parser.add_argument('--error_neg_index_offset', type=int, default=2500,
                        help='x-index for negative-class samples starts at this offset (positive class uses 0,1,…)')
    parser.add_argument('--error_sample_dir', type=str, default='20260511error_output/sample',
                        help='dir under project root for FP/FN top-k samples (far from P=0.5), or absolute path')
    parser.add_argument('--error_sample_top_n', type=int, default=10,
                        help='how many FP and FN images to copy into sample dir')
    parser.add_argument('--no_error_cm_montage', action='store_true',
                        help='disable CM montage PNG (default: build when --save_error_images)')
    parser.add_argument('--error_cm_montage_rows', type=int, default=20,
                        help='montage grid rows (default 20; with cols 10 => 200 cells)')
    parser.add_argument('--error_cm_montage_cols', type=int, default=10,
                        help='montage grid columns (default 10)')
    parser.add_argument('--error_cm_montage_cell_size', type=int, default=128,
                        help='square cell size in px for CM montage')
    parser.add_argument('--error_cm_montage_seed', type=int, default=42,
                        help='RNG seed for stratified sampling')
    parser.add_argument('--error_cm_montage_filename', type=str, default='cm_montage_20x10.png',
                        help='filename under error_output_dir')
    parser.add_argument('--no_error_cm_quad', action='store_true',
                        help='disable per-class 2x2 quad PNGs (cm_quad_TP.png, ...)')
    parser.add_argument('--error_cm_quad_filename_prefix', type=str, default='cm_quad',
                        help='quad outputs: {prefix}_TP.png, _FP.png, _FN.png, _TN.png')
    parser.add_argument('--error_fn_montage_n', type=int, default=35,
                        help='first N FN: original + model-input montage (7x10 when rows=7 half_cols=5)')
    parser.add_argument('--no_error_fn_montage', action='store_true',
                        help='disable fn_original_vs_input_montage.png/pdf')
    parser.add_argument('--error_fn_montage_cell_size', type=int, default=128,
                        help='cell size in px for FN original vs input montage')
    parser.add_argument('--positive_class_index', type=int, default=0,
                        help='ImageFolder class index for fire/有火 (0 or 1; must match data/class order)')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)
    if getattr(args, 'no_bowfire_flat_train', False):
        args.bowfire_allow_flat_train = False
    elif not hasattr(args, 'bowfire_allow_flat_train'):
        args.bowfire_allow_flat_train = True
    if getattr(args, 'no_bowfire_flat_test', False):
        args.bowfire_allow_flat_test = False
    elif not hasattr(args, 'bowfire_allow_flat_test'):
        args.bowfire_allow_flat_test = True

    main(args)

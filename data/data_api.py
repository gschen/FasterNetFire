import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.utils import str2list
from data.custom_imagenet_data import custom_ImagenetDataModule
from data.custom_imagenet_data import build_imagenet_transform


__all__ = ['LitDataModule']


def LitDataModule(hparams):
    dm =None
    CLASS_NAMES = None
    # 小数据集（如 BoWFire）在 DDP + 大 batch 下若 drop_last=True 容易出现 dataloader 长度为 0
    drop_last = bool(getattr(hparams, "drop_last", False))
    batch_size = hparams.batch_size
    batch_size_eva = hparams.batch_size_eva

    dataset_name = str(hparams.dataset_name).strip().lower()
    _bowfire_names = {
        "bowfiredataset",
        "bowfire",
        "bow fire dataset",
    }
    _layout = getattr(hparams, "dataset_layout", None)
    if _layout is None or str(_layout).strip() == "":
        _layout = "bowfire" if dataset_name in _bowfire_names else "auto"
    else:
        _layout = str(_layout).strip().lower()
    if dataset_name in {
        "fd-dataset",
        "mivia",
        "mivia fire detection dataset",
        *_bowfire_names,
    }:
        if dataset_name in _bowfire_names:
            # BoWFire 规模较小，强制不丢弃最后一个 batch，避免 0-step 训练
            drop_last = False
        dm = custom_ImagenetDataModule(
            image_size=hparams.image_size,
            data_dir=hparams.data_dir,
            train_transforms=build_imagenet_transform(is_train=True, args=hparams, image_size=hparams.image_size),
            train_transforms_multi_scale=None if hparams.multi_scale is None else build_imagenet_transform(
                is_train=True, args=hparams, image_size=int(hparams.multi_scale.split('_')[0])),
            scaling_epoch=None if hparams.multi_scale is None else int(hparams.multi_scale.split('_')[1]),
            val_transforms=build_imagenet_transform(is_train=False, args=hparams, image_size=hparams.image_size),
            num_workers=hparams.num_workers,
            pin_memory=hparams.pin_memory,
            # dist_eval= True if len(str2list(hparams.gpus))>1 else False,
            batch_size=batch_size,
            batch_size_eva=batch_size_eva,
            drop_last=drop_last,
            split_seed=int(getattr(hparams, 'data_split_seed', 42)),
            split_ratio=(
                float(getattr(hparams, 'data_split_train_ratio', 0.7)),
                float(getattr(hparams, 'data_split_val_ratio', 0.2)),
                float(getattr(hparams, 'data_split_test_ratio', 0.1)),
            ),
            dataset_layout=_layout,
            bowfire_allow_flat_train=bool(getattr(hparams, 'bowfire_allow_flat_train', True)),
            bowfire_allow_flat_test=bool(getattr(hparams, 'bowfire_allow_flat_test', True)),
            bowfire_test_only=bool(getattr(hparams, 'bowfire_test_only', False)),
            nofire_extra_enable=bool(getattr(hparams, 'nofire_extra_enable', True)),
            nofire_extra_categories=getattr(
                hparams, 'nofire_extra_categories', ["日出", "日落", "夕阳", "晚霞", "火烧云"]
            ),
            nofire_extra_start=int(getattr(hparams, 'nofire_extra_start', 1)),
            nofire_extra_end=int(getattr(hparams, 'nofire_extra_end', 1000)),
        )
    else:
        print("Invalid dataset name, exiting...")
        exit()

    return dm, CLASS_NAMES
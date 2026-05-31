# type: ignore[override]
import os
import random
import re
from typing import Any, Callable, Optional, List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from utils.utils import *

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

# BoWFire：扩展后缀 + 不依赖 ImageFolder 对「一级子目录」的严格检测
_IMG_EXTS = (
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".gif",
)


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMG_EXTS


def _immediate_subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )


def _count_images_under(path: str) -> int:
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            if _is_image_file(f):
                n += 1
    return n


def _bowfire_class_semantic(dirname: str) -> Optional[str]:
    """将文件夹名映射为 fire / nofire（语义），无法识别则返回 None。"""
    n = dirname.lower().replace("-", "_").strip()
    if n in ("fire", "flame", "positive", "pos"):
        return "fire"
    if n in ("no_fire", "nofire", "non_fire", "negative", "neg", "background", "bg", "not_fire"):
        return "nofire"
    return None


def _bowfire_infer_label_from_basename(basename: str) -> Optional[str]:
    """扁平目录时根据文件名猜测标签（先判非火再判火，避免 no_fire 中含 fire 子串）。"""
    low = basename.lower()
    if any(
        x in low
        for x in (
            "no_fire",
            "nofire",
            "non_fire",
            "not_fire",
            "neg_",
            "negative",
            "background",
            "_bg_",
            "-bg-",
            "normal",
            "smoke",
            "nonfire",
        )
    ):
        return "nofire"
    if "fire" in low or "flame" in low:
        return "fire"
    return None


class _BowFireBinaryFolder:
    """行为接近 ImageFolder：samples, classes, class_to_idx, loader。"""

    def __init__(self, samples: List[Tuple[str, int]], classes: List[str]):
        from torchvision.datasets.folder import default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.loader = default_loader
        self.imgs = self.samples


def _bowfire_build_from_class_subdirs(parent_dir: str) -> _BowFireBinaryFolder:
    """从 parent_dir 下一级子目录收集两类图片（每类至少一张图）。"""
    subdirs = _immediate_subdirs(parent_dir)
    with_img = [d for d in subdirs if _count_images_under(os.path.join(parent_dir, d)) > 0]
    if len(with_img) < 2:
        raise FileNotFoundError(
            f"BoWFire: 在 {parent_dir} 需要至少 2 个「含图片」的一级子文件夹；"
            f"当前子目录: {subdirs}"
        )
    fire_dirs = [d for d in with_img if _bowfire_class_semantic(d) == "fire"]
    nf_dirs = [d for d in with_img if _bowfire_class_semantic(d) == "nofire"]
    unknown = [d for d in with_img if d not in fire_dirs and d not in nf_dirs]
    if len(fire_dirs) == 1 and len(nf_dirs) == 1:
        class_names = sorted([fire_dirs[0], nf_dirs[0]])
    elif len(with_img) == 2:
        class_names = sorted(with_img)
    else:
        raise ValueError(
            f"BoWFire: 无法唯一识别 fire / 非火 文件夹。含图子目录={with_img}；"
            f"识别为 fire={fire_dirs} no_fire={nf_dirs} 未识别={unknown}"
        )
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples: List[Tuple[str, int]] = []
    for cname in class_names:
        cdir = os.path.join(parent_dir, cname)
        for r, _, files in os.walk(cdir):
            for f in files:
                if _is_image_file(f):
                    samples.append((os.path.join(r, f), class_to_idx[cname]))
    if not samples:
        raise RuntimeError(f"BoWFire: 未在 {parent_dir} 收集到任何图片")
    return _BowFireBinaryFolder(samples, class_names)


def _bowfire_build_flat_dir(parent_dir: str) -> _BowFireBinaryFolder:
    """parent_dir 下无子目录，所有图片在同一目录；按文件名推断 fire / nofire。"""
    files = [
        os.path.join(parent_dir, f)
        for f in os.listdir(parent_dir)
        if os.path.isfile(os.path.join(parent_dir, f)) and _is_image_file(f)
    ]
    if not files:
        raise FileNotFoundError(f"BoWFire: 扁平目录 {parent_dir} 中未找到图片文件")
    by_sem: dict = {"fire": [], "nofire": []}
    skipped = 0
    for p in files:
        sem = _bowfire_infer_label_from_basename(os.path.basename(p))
        if sem is None:
            skipped += 1
            continue
        by_sem[sem].append(p)
    if not by_sem["fire"] or not by_sem["nofire"]:
        raise ValueError(
            f"BoWFire: 扁平目录 {parent_dir} 无法拆成两类（fire={len(by_sem['fire'])}, "
            f"nofire={len(by_sem['nofire'])}, 无法从文件名推断的={skipped}）。"
            f"请改为 train/fire 与 train/no_fire 子目录，或保证文件名含 fire / no_fire 等关键词。"
        )
    class_names = ["fire", "no_fire"]
    samples = [(p, 0) for p in by_sem["fire"]] + [(p, 1) for p in by_sem["nofire"]]
    return _BowFireBinaryFolder(samples, class_names)


def _bowfire_load_split_dir(path: str, allow_flat: bool) -> _BowFireBinaryFolder:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"BoWFire: 目录不存在: {path}")
    subs = _immediate_subdirs(path)
    subs_with_img = [d for d in subs if _count_images_under(os.path.join(path, d)) > 0]
    root_files = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and _is_image_file(f)
    ]
    if len(subs_with_img) >= 2:
        try:
            return _bowfire_build_from_class_subdirs(path)
        except ValueError:
            if allow_flat and root_files:
                return _bowfire_build_flat_dir(path)
            raise
    if allow_flat and root_files:
        return _bowfire_build_flat_dir(path)
    return _bowfire_build_from_class_subdirs(path)


def _bowfire_align_test_to_train(ref: _BowFireBinaryFolder, test: _BowFireBinaryFolder) -> List[Tuple[str, int]]:
    ref_by_sem: dict = {}
    for cname in ref.classes:
        sem = _bowfire_class_semantic(cname)
        if sem is None:
            raise ValueError(
                f"BoWFire: train 类别文件夹名无法映射为 fire/非火: {cname!r}，当前 classes={ref.classes}"
            )
        if sem in ref_by_sem:
            raise ValueError(f"BoWFire: train 类别语义重复: {ref.classes}")
        ref_by_sem[sem] = ref.class_to_idx[cname]
    need = {"fire", "nofire"}
    if set(ref_by_sem.keys()) != need:
        raise ValueError(f"BoWFire: train 需同时包含 fire 与 非火 两类，当前语义映射={ref_by_sem}")
    out: List[Tuple[str, int]] = []
    for p, y in test.samples:
        tname = test.classes[y]
        sem = _bowfire_class_semantic(tname)
        if sem is None:
            raise ValueError(
                f"BoWFire: test 类别文件夹名无法映射: {tname!r}，当前 classes={test.classes}"
            )
        out.append((p, ref_by_sem[sem]))
    return out


class _ImageFolderFromSamples(Dataset):
    """A lightweight ImageFolder-compatible dataset from selected sample tuples."""

    def __init__(self, base_dataset: Any, samples: List[Tuple[str, int]], transform=None):
        self.base = base_dataset
        self.samples = list(samples)
        self.imgs = self.samples
        self.targets = [s[1] for s in self.samples]
        self.classes = getattr(base_dataset, "classes", None)
        self.class_to_idx = getattr(base_dataset, "class_to_idx", None)
        self.transform = transform
        self.target_transform = None
        self.loader = base_dataset.loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class custom_ImagenetDataModule(LightningDataModule):
    """
    The train set is the imagenet train.
    The val/test set are the official imagenet validation set.

    """

    name = "FD-dataset"

    def __init__(
        self,
        data_dir: str,
        meta_dir: Optional[str] = None,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        batch_size_eva: int = 32,
        # dist_eval: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transforms_multi_scale = None,
        scaling_epoch = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            meta_dir: path to meta.bin file
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        # Pop custom args before calling LightningDataModule.__init__
        split_seed = int(kwargs.pop("split_seed", 42))
        split_ratio = kwargs.pop("split_ratio", (0.7, 0.2, 0.1))
        self.dataset_layout = str(kwargs.pop("dataset_layout", "auto")).strip().lower()
        self.bowfire_allow_flat_train = bool(kwargs.pop("bowfire_allow_flat_train", True))
        self.bowfire_allow_flat_test = bool(kwargs.pop("bowfire_allow_flat_test", True))
        self.bowfire_test_only = bool(kwargs.pop("bowfire_test_only", False))
        self.nofire_extra_enable = bool(kwargs.pop("nofire_extra_enable", True))
        self.nofire_extra_categories = kwargs.pop(
            "nofire_extra_categories", ["日出", "日落", "夕阳", "晚霞", "火烧云"]
        )
        self.nofire_extra_start = int(kwargs.pop("nofire_extra_start", 1))
        self.nofire_extra_end = int(kwargs.pop("nofire_extra_end", 1000))
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use ImageNet dataset loaded from `torchvision` which is not installed yet."
            )

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.batch_size = batch_size
        self.batch_size_eva = batch_size_eva
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = 1281167
        self.num_tasks = get_world_size()
        self.global_rank = get_rank()
        self.train_transforms_multi_scale = train_transforms_multi_scale
        self.scaling_epoch = scaling_epoch
        self.split_seed = split_seed
        if not isinstance(split_ratio, (tuple, list)) or len(split_ratio) != 3:
            raise ValueError(f"split_ratio must be a tuple/list of 3 floats, got: {split_ratio}")
        self.split_ratio = tuple(float(x) for x in split_ratio)
        # self.dist_eval = dist_eval

        # 兼容两类目录:
        # 1) data_dir/data/{train,validation,test}
        # 2) data_dir/{fire,no_fire} (自动按 split_ratio 切分)
        self._split_root = data_dir
        data_subdir = os.path.join(data_dir, "data")
        self.data_dir = data_subdir if os.path.isdir(data_subdir) else data_dir

    @property
    def num_classes(self) -> int:
        return 2

    def _bowfire_root(self) -> str:
        return self._split_root

    def _is_bowfire_layout(self) -> bool:
        if self.dataset_layout == "bowfire":
            return os.path.isdir(os.path.join(self._bowfire_root(), "dataset", "img"))
        if self.dataset_layout not in ("auto", ""):
            return False
        root = self._bowfire_root()
        test_img = os.path.join(root, "dataset", "img")
        val_dir = os.path.join(self.data_dir, "validation")
        # 仅当存在 dataset/img 且非标准 ImageNet 式 data/validation 布局时视为 BoWFire（train/ 目录可存在但不再使用）
        if os.path.isdir(test_img) and not os.path.isdir(val_dir):
            return True
        return False

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.

        .. warning:: Please download imagenet on your own first.
        To get imagenet:
        1. download yourself from http://www.image-net.org/challenges/LSVRC/2012/downloads
        2. download the devkit (ILSVRC2012_devkit_t12.tar.gz)
        """
        if self._is_bowfire_layout():
            root = self._bowfire_root()
            test_img = os.path.join(root, "dataset", "img")
            if not os.path.isdir(test_img):
                raise FileNotFoundError(f"BoWFire: 需要目录 dataset/img/: {test_img}")
            gt_dir = os.path.join(root, "dataset", "gt")
            if not os.path.isdir(gt_dir):
                print(f"[BoWFire] 未找到分割标注目录（可选）: {gt_dir}")
            return
        if os.path.isdir(os.path.join(self.data_dir, "train")):
            self._verify_splits(self.data_dir, "train")
            self._verify_splits(self.data_dir, "validation")
            return
        # flat class-folder mode: /data/MIVIA/fire, /data/MIVIA/no_fire
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")
        class_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        if len(class_dirs) < 2:
            raise FileNotFoundError(
                f"Expect at least 2 class folders under {self.data_dir}, got: {class_dirs}"
            )

    def _make_splits_from_flat_root(self, train_transforms, val_transforms):
        base = datasets.ImageFolder(self.data_dir, transform=None)
        by_class = {}
        for p, y in base.samples:
            by_class.setdefault(y, []).append((p, y))

        tr, vr, te = self.split_ratio
        if tr <= 0 or vr <= 0 or te <= 0 or abs((tr + vr + te) - 1.0) > 1e-6:
            raise ValueError(f"split_ratio must sum to 1.0, got {self.split_ratio}")

        rng = random.Random(self.split_seed)
        train_samples, val_samples, test_samples = [], [], []
        for cls in sorted(by_class.keys()):
            items = list(by_class[cls])
            rng.shuffle(items)
            n = len(items)
            n_train = int(n * tr)
            n_val = int(n * vr)
            n_test = n - n_train - n_val
            # 避免极端情况下某一分割为空
            if n >= 3:
                if n_train == 0:
                    n_train = 1
                if n_val == 0:
                    n_val = 1
                n_test = n - n_train - n_val
            train_samples.extend(items[:n_train])
            val_samples.extend(items[n_train:n_train + n_val])
            test_samples.extend(items[n_train + n_val:n_train + n_val + n_test])

        self.dataset_train = _ImageFolderFromSamples(base, train_samples, transform=train_transforms)
        self.dataset_val = _ImageFolderFromSamples(base, val_samples, transform=val_transforms)
        self.dataset_test = _ImageFolderFromSamples(base, test_samples, transform=val_transforms)

        if self.train_transforms_multi_scale is not None:
            self.dataset_train_multi_scale = _ImageFolderFromSamples(
                base, train_samples, transform=self.train_transforms_multi_scale
            )
        else:
            self.dataset_train_multi_scale = None

    def _parse_nofire_extra_categories(self) -> List[str]:
        v = self.nofire_extra_categories
        if isinstance(v, str):
            # support "日出,日落,夕阳" in yaml/cli
            return [x.strip() for x in v.split(",") if x.strip()]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    @staticmethod
    def _parse_augmented_nofire_filename(path: str):
        """
        Parse "<category>_<index>.jpg" like "日出_1.jpg".
        Return (category, index) or (None, None) when unmatched.
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r"^(.+)_([0-9]+)$", stem)
        if not m:
            return None, None
        return m.group(1), int(m.group(2))

    def _filter_train_nofire_extras(self, dataset):
        """
        Filter only augmented no_fire images by config:
        - nofire_extra_enable: whether to include augmented no_fire images
        - nofire_extra_categories: category name whitelist
        - nofire_extra_start/end: index range (inclusive)
        """
        categories = set(self._parse_nofire_extra_categories())
        start_i = min(self.nofire_extra_start, self.nofire_extra_end)
        end_i = max(self.nofire_extra_start, self.nofire_extra_end)
        classes = getattr(dataset, "classes", [])
        class_to_idx = getattr(dataset, "class_to_idx", {})
        nofire_idx_set = {
            class_to_idx[c]
            for c in classes
            if _bowfire_class_semantic(c) == "nofire"
        }
        # fallback for common FD-dataset naming
        for c in classes:
            cl = c.lower()
            if cl in {"no_fire", "nofire", "non_fire", "background"} and c in class_to_idx:
                nofire_idx_set.add(class_to_idx[c])

        if not nofire_idx_set:
            return dataset

        kept = []
        removed_aug = 0
        total_aug = 0
        for p, y in dataset.samples:
            if y not in nofire_idx_set:
                kept.append((p, y))
                continue
            cat, idx = self._parse_augmented_nofire_filename(p)
            if cat is None:
                # not an augmented sample filename, always keep
                kept.append((p, y))
                continue
            if categories and cat not in categories:
                kept.append((p, y))
                continue
            total_aug += 1
            in_range = start_i <= idx <= end_i
            use_this = bool(self.nofire_extra_enable) and in_range
            if use_this:
                kept.append((p, y))
            else:
                removed_aug += 1

        if removed_aug > 0:
            print(
                f"[FD no_fire extras] kept {total_aug - removed_aug}/{total_aug} augmented samples "
                f"(enable={self.nofire_extra_enable}, range={start_i}-{end_i}, categories={sorted(categories)})"
            )

        if len(kept) == len(dataset.samples):
            return dataset
        return _ImageFolderFromSamples(dataset, kept, transform=getattr(dataset, "transform", None))

    def _setup_bowfire(self, train_transforms, val_transforms) -> None:
        """BoWFire: 使用 dataset/img/，可按需切分或直接全量测试。"""
        root = self._bowfire_root()
        img_root = os.path.join(root, "dataset", "img")
        allow_flat = self.bowfire_allow_flat_train or self.bowfire_allow_flat_test
        base = _bowfire_load_split_dir(img_root, allow_flat)
        if self.bowfire_test_only:
            self.dataset_test = _ImageFolderFromSamples(base, base.samples, transform=val_transforms)
            self.dataset_train = None
            self.dataset_val = None
            self.dataset_train_multi_scale = None
            c0 = sum(1 for _, y in base.samples if y == 0)
            c1 = sum(1 for _, y in base.samples if y == 1)
            print(
                f"[BoWFire] TEST-ONLY: use all images under dataset/img as test set: "
                f"test={len(base.samples)} (class0={c0}, class1={c1})"
            )
            return

        tr, vr, te = self.split_ratio
        if tr <= 0 or vr <= 0 or te <= 0 or abs((tr + vr + te) - 1.0) > 1e-6:
            raise ValueError(
                f"BoWFire: data_split_train/val/test_ratio 三项须为正且和为 1.0，当前 {self.split_ratio}"
            )
        by_class = {}
        for p, y in base.samples:
            by_class.setdefault(y, []).append((p, y))
        rng = random.Random(self.split_seed)
        train_samples, val_samples, test_samples = [], [], []
        for cls in sorted(by_class.keys()):
            items = list(by_class[cls])
            rng.shuffle(items)
            n = len(items)
            n_train = int(n * tr)
            n_val = int(n * vr)
            n_test = n - n_train - n_val
            if n >= 3:
                if n_train == 0:
                    n_train = 1
                if n_val == 0:
                    n_val = 1
                n_test = n - n_train - n_val
            train_samples.extend(items[:n_train])
            val_samples.extend(items[n_train : n_train + n_val])
            test_samples.extend(items[n_train + n_val : n_train + n_val + n_test])
        self.dataset_train = _ImageFolderFromSamples(base, train_samples, transform=train_transforms)
        self.dataset_val = _ImageFolderFromSamples(base, val_samples, transform=val_transforms)
        self.dataset_test = _ImageFolderFromSamples(base, test_samples, transform=val_transforms)
        if self.train_transforms_multi_scale is not None:
            self.dataset_train_multi_scale = _ImageFolderFromSamples(
                base, train_samples, transform=self.train_transforms_multi_scale
            )
        else:
            self.dataset_train_multi_scale = None

        def _cnt(samples):
            c0 = sum(1 for _, y in samples if y == 0)
            c1 = sum(1 for _, y in samples if y == 1)
            return c0, c1

        tr0, tr1 = _cnt(train_samples)
        va0, va1 = _cnt(val_samples)
        te0, te1 = _cnt(test_samples)
        print(
            f"[BoWFire] 仅 dataset/img/ | train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} "
            f"| ratio={tr}:{vr}:{te}"
        )
        print(
            f"[BoWFire] class idx0={base.classes[0]} idx1={base.classes[1]} | "
            f"train=({tr0},{tr1}) val=({va0},{va1}) test=({te0},{te1})"
        )

    def _print_dataset_stats(self, split: str) -> None:
        """Print dataset size and per-class counts for verification."""
        ds = getattr(self, f"dataset_{split}", None)
        if ds is None:
            print(f"[Data] {split} dataset is None")
            return

        total = len(ds)
        classes = getattr(ds, "classes", None)
        targets = getattr(ds, "targets", None)
        if targets is None and hasattr(ds, "samples"):
            targets = [y for _, y in ds.samples]
        if targets is None:
            print(f"[Data] {split} samples: {total}")
            return

        by_cls = {}
        for y in targets:
            by_cls[int(y)] = by_cls.get(int(y), 0) + 1
        if classes is not None:
            detail = ", ".join(
                f"{i}:{classes[i]}={by_cls.get(i, 0)}" for i in range(len(classes))
            )
        else:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(by_cls.items()))
        print(f"[Data] {split} samples: {total} | {detail}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms

        if self._is_bowfire_layout():
            self._setup_bowfire(train_transforms, val_transforms)
            if stage == "fit" or stage is None:
                self._print_dataset_stats("train")
                self._print_dataset_stats("val")
            if stage == "test" or stage is None:
                self._print_dataset_stats("test")
            return

        use_pre_split = os.path.isdir(os.path.join(self.data_dir, "train"))
        if use_pre_split:
            if stage == "fit" or stage is None:
                self.dataset_train = datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=train_transforms)
                self.dataset_val = datasets.ImageFolder(os.path.join(self.data_dir, "validation"), transform=val_transforms)
                self.dataset_train = self._filter_train_nofire_extras(self.dataset_train)
                if self.train_transforms_multi_scale is not None:
                    self.dataset_train_multi_scale = datasets.ImageFolder(
                        os.path.join(self.data_dir, "train"), transform=self.train_transforms_multi_scale
                    )
                    self.dataset_train_multi_scale = self._filter_train_nofire_extras(self.dataset_train_multi_scale)
                else:
                    self.dataset_train_multi_scale = None
                self._print_dataset_stats("train")
                self._print_dataset_stats("val")
            if stage == "test" or stage is None:
                self.dataset_test = datasets.ImageFolder(os.path.join(self.data_dir, "test"), transform=val_transforms)
                self._print_dataset_stats("test")
        else:
            # flat-root mode: create all splits in one go for consistency
            self._make_splits_from_flat_root(train_transforms, val_transforms)
            if stage == "fit" or stage is None:
                self._print_dataset_stats("train")
                self._print_dataset_stats("val")
            if stage == "test" or stage is None:
                self._print_dataset_stats("test")

    def train_dataloader(self) -> DataLoader:
        if self.dataset_train_multi_scale is not None and \
                self.trainer.current_epoch < self.scaling_epoch:
            dataset = self.dataset_train_multi_scale
            print("load dataset_train_multi_scale")
        else:
            dataset = self.dataset_train
            print("load dataset_train")

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader: DataLoader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_eva,
            # persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Uses the validation split of imagenet2012 for testing."""
        loader: DataLoader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_eva,
            # persistent_workers=True,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory
        )
        return loader

    def train_transform(self) -> Callable:
        preprocessing = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:

        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing


def build_imagenet_transform(is_train, args, image_size):
    resize_im = image_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            # use_prefetcher=args.use_prefetcher,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                image_size, padding=4)
        return transform

    t = []
    eval_resize_only = bool(getattr(args, 'eval_resize_only', False))
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if image_size >= 384 or eval_resize_only:
            t.append(
                transforms.Resize((image_size, image_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            if eval_resize_only and image_size < 384:
                print(f"Eval resize-only enabled: directly resize to {image_size}x{image_size} (no center crop).")
            else:
                print(f"Warping {image_size} size input images...")
        else:
            # size = int((256 / 224) * image_size)
            size = int(1.0*image_size/args.test_crop_ratio)
            t.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
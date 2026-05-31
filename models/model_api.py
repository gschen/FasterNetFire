import os
import random
import re
import shutil
import sys
import inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, parentdir)

from torch.optim.lr_scheduler import *
from models import *
from utils.utils import *
import torch
import pytorch_lightning as pl

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import DistillationLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from PIL import Image
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_criterion(args):
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def build_mixup_fn(args, num_classes):
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)
    return mixup_fn


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        from .fasternetfire import FasterNetFire
        self.model = FasterNetFire(
            mlp_ratio=hparams.mlp_ratio,
            embed_dim=hparams.embed_dim,
            depths=hparams.depths,
            pretrained=hparams.pretrained,
            n_div=hparams.n_div,
            feature_dim=hparams.feature_dim,
            patch_size=hparams.patch_size,
            patch_stride=hparams.patch_stride,
            patch_size2=hparams.patch_size2,
            patch_stride2=hparams.patch_stride2,
            num_classes=num_classes,
            layer_scale_init_value=hparams.layer_scale_init_value,
            drop_path_rate=hparams.drop_path_rate,
            norm_layer=hparams.norm_layer,
            act_layer=hparams.act_layer,
            pconv_fw_type=getattr(hparams, 'pconv_fw_type', 'split_cat'),
        )

        base_criterion = build_criterion(hparams)
        self.distillation_type = hparams.distillation_type
        if hparams.distillation_type == 'none':
            self.criterion = base_criterion
        else:
            # assert hparams.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {hparams.teacher_model}")
            teacher_model = create_model(
                hparams.teacher_model,
                pretrained=True,
                num_classes=num_classes,
                global_pool='avg',
            )
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            self.criterion = DistillationLoss(base_criterion,
                                              teacher_model,
                                              hparams.distillation_type,
                                              hparams.distillation_alpha,
                                              hparams.distillation_tau
                                              )
        self.criterion_eva = torch.nn.CrossEntropyLoss()
        self.mixup_fn = build_mixup_fn(hparams, num_classes)
        
        # 初始化torchmetrics指标
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.test_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        
        # 每个类别的指标
        self.val_precision_per_class = Precision(task='multiclass', num_classes=num_classes, average='none')
        self.val_recall_per_class = Recall(task='multiclass', num_classes=num_classes, average='none')
        self.val_f1_per_class = F1Score(task='multiclass', num_classes=num_classes, average='none')
        
        self.test_precision_per_class = Precision(task='multiclass', num_classes=num_classes, average='none')
        self.test_recall_per_class = Recall(task='multiclass', num_classes=num_classes, average='none')
        self.test_f1_per_class = F1Score(task='multiclass', num_classes=num_classes, average='none')

    def _quadrant_label_prob_pos(self, y_true, y_pred, prob_fire):
        """四象限标签：纵轴为 P(有火)，下标由 positive_class_index 指定。

        - 预测正确：P(fire)>0.5 → TP；P(fire)≤0.5 → TN
        - 预测错误：P(fire)>0.5 → FP；P(fire)≤0.5 → FN
        """
        correct = (y_true == y_pred)
        if correct:
            return 'TP' if prob_fire > 0.5 else 'TN'
        return 'FP' if prob_fire > 0.5 else 'FN'

    def _save_original_image_jpg(self, src_path, dst_path):
        with Image.open(src_path) as im:
            im = im.convert('RGB')
            im.save(dst_path, 'JPEG', quality=95)

    def _save_error_analysis_samples(self, imgs, labels, logits):
        """error_inf.txt 与散点：纵坐标均为 P(有火)。错分图从 dataset 原始路径保存为 JPEG（非 224 输入）。

        命名：{类别}_{P(fire)}_{序号}.jpg；仅 FP/FN 落盘。
        """
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        out_dir = self.hparams.error_output_dir
        fire_idx = int(getattr(self.hparams, 'positive_class_index', 0))
        prob_fire = probs[:, fire_idx]
        dataset_test = self.trainer.datamodule.dataset_test

        neg_off = int(getattr(self.hparams, 'error_neg_index_offset', 2500))
        for i in range(imgs.shape[0]):
            flat_idx = self._test_flat_idx
            self._test_flat_idx += 1
            y = int(labels[i].item())
            if y == fire_idx:
                gidx = self._idx_pos
                self._idx_pos += 1
            else:
                gidx = neg_off + self._idx_neg
                self._idx_neg += 1
            p = int(pred[i].item())
            score = float(prob_fire[i].item())
            cm = self._quadrant_label_prob_pos(y, p, score)
            fname = f'{cm}_{score:.4f}_{gidx}.jpg'
            self._all_inf_lines.append(fname)
            self._scatter_by_cm[cm].append((gidx, score))
            self._flat_by_cm[cm].append(flat_idx)
            if cm in ('FP', 'FN'):
                fpath = os.path.join(out_dir, fname)
                src_path = dataset_test.samples[flat_idx][0]
                self._save_original_image_jpg(src_path, fpath)
                if cm == 'FN':
                    n_fn_cap = int(getattr(self.hparams, 'error_fn_montage_n', 35))
                    if len(self._fn_showcase_rows) < n_fn_cap:
                        self._fn_showcase_rows.append(
                            (src_path, imgs[i].detach().float().cpu().clone())
                        )
                dist = abs(score - 0.5)
                if cm == 'FP':
                    self._fp_dist_paths.append((dist, fpath))
                else:
                    self._fn_dist_paths.append((dist, fpath))

    def _resolve_sample_dir(self):
        p = getattr(self.hparams, 'error_sample_dir', 'sample')
        if os.path.isabs(p):
            return p
        return os.path.join(parentdir, p)

    def _parse_p_fire_from_sample_name(self, path):
        # FP_0.9234_2529.jpg
        m = re.match(r'^[A-Z]{2}_([0-9]+(?:\.[0-9]+)?)_\d+\.jpg$', os.path.basename(path), re.I)
        return float(m.group(1)) if m else 0.0

    def _build_sample_montage(self, fp_sorted, fn_sorted, sample_dir):
        """4 rows x 5 cols; col0 FP/FN; rows 0–1 FP, 2–3 FN; P(fire) under each image."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        def resolve_in_sample(src):
            if not src:
                return None
            local = os.path.join(sample_dir, os.path.basename(src))
            if os.path.isfile(local):
                return local
            return src if os.path.isfile(src) else None

        fp_paths = [resolve_in_sample(p) for _, p in fp_sorted[:10]]
        fn_paths = [resolve_in_sample(p) for _, p in fn_sorted[:10]]
        while len(fp_paths) < 10:
            fp_paths.append(None)
        while len(fn_paths) < 10:
            fn_paths.append(None)

        rc = {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 11,
            'axes.unicode_minus': False,
        }
        box_hw = 3.0 / 4.0
        with plt.rc_context(rc=rc):
            fig = plt.figure(figsize=(14, 8.5))
            gs = fig.add_gridspec(
                4, 6,
                width_ratios=[0.28, 1, 1, 1, 1, 1],
                height_ratios=[1, 1, 1, 1],
                hspace=0.14,
                wspace=0.08,
                left=0.04, right=0.98, top=0.96, bottom=0.05,
            )
            for r in range(4):
                ax_left = fig.add_subplot(gs[r, 0])
                ax_left.axis('off')
                cat = 'FP' if r < 2 else 'FN'
                ax_left.text(
                    0.5, 0.5, cat, ha='center', va='center',
                    fontsize=16, fontweight='semibold', transform=ax_left.transAxes,
                )
                if r < 2:
                    row_paths = fp_paths[r * 5:(r + 1) * 5]
                else:
                    row_paths = fn_paths[(r - 2) * 5:(r - 1) * 5]
                for c in range(5):
                    ax = fig.add_subplot(gs[r, c + 1])
                    ax.axis('off')
                    pth = row_paths[c]
                    if pth and os.path.isfile(pth):
                        im = np.asarray(Image.open(pth).convert('RGB'))
                        ax.imshow(im, aspect='auto')
                        ax.set_box_aspect(box_hw)
                        prob = self._parse_p_fire_from_sample_name(pth)
                        ax.text(
                            0.5, -0.05, f'P(fire)={prob:.4f}',
                            transform=ax.transAxes, ha='center', va='top',
                            fontsize=15,
                        )
                    else:
                        ax.imshow(
                            np.ones((3, 4, 3), dtype=np.float32) * 0.94,
                            aspect='auto',
                        )
                        ax.set_box_aspect(box_hw)
                        ax.text(0.5, 0.5, '—', ha='center', va='center', transform=ax.transAxes)

            out_png = os.path.join(sample_dir, 'sample_montage.png')
            save_figure_png_and_pdf(fig, out_png, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

    def _export_sample_far_from_half(self):
        """Copy top-N FP and top-N FN (farthest from P=0.5) into sample/ with same names as error_output."""
        top_n = int(getattr(self.hparams, 'error_sample_top_n', 10))
        sample_dir = self._resolve_sample_dir()
        os.makedirs(sample_dir, exist_ok=True)
        fp_sorted = sorted(self._fp_dist_paths, key=lambda x: -x[0])[:top_n]
        fn_sorted = sorted(self._fn_dist_paths, key=lambda x: -x[0])[:top_n]
        for _, src in fp_sorted:
            if os.path.isfile(src):
                dst = os.path.join(sample_dir, os.path.basename(src))
                shutil.copy2(src, dst)
        for _, src in fn_sorted:
            if os.path.isfile(src):
                dst = os.path.join(sample_dir, os.path.basename(src))
                shutil.copy2(src, dst)
        if fp_sorted or fn_sorted:
            self._build_sample_montage(fp_sorted, fn_sorted, sample_dir)

    def _sample_flat_indices_cm_montage(self, rng, n_total):
        """分层每类最多 n_total//4 张；不足则从剩余样本中补足，仍不足则放回抽样。

        返回 (flat_indices, by_cm)：
        - flat_indices：打乱后用于大图拼贴的顺序；
        - by_cm：本次抽中的 flat_idx 按 TP/FP/FN/TN 分组（与打乱顺序无关）。
        """
        order = ('TP', 'FP', 'FN', 'TN')
        per = n_total // 4
        pools = {cm: list(self._flat_by_cm[cm]) for cm in order}
        for cm in order:
            rng.shuffle(pools[cm])
        pairs = []
        for cm in order:
            pool = pools[cm]
            take = min(per, len(pool))
            for i in range(take):
                pairs.append((pool[i], cm))
            pools[cm] = pool[take:]
        deficit = n_total - len(pairs)
        if deficit > 0:
            rest = []
            for cm in order:
                for idx in pools[cm]:
                    rest.append((idx, cm))
            rng.shuffle(rest)
            if len(rest) >= deficit:
                pairs.extend(rest[:deficit])
            else:
                pairs.extend(rest)
                all_pairs = []
                for cm in order:
                    for idx in self._flat_by_cm[cm]:
                        all_pairs.append((idx, cm))
                if not all_pairs:
                    pairs = pairs[:n_total]
                    by_cm = {k: [] for k in order}
                    for idx, cm in pairs:
                        by_cm[cm].append(idx)
                    rng.shuffle(pairs)
                    return [p[0] for p in pairs], by_cm
                while len(pairs) < n_total:
                    pairs.append(rng.choice(all_pairs))
        pairs = pairs[:n_total]
        by_cm = {k: [] for k in order}
        for idx, cm in pairs:
            by_cm[cm].append(idx)
        rng.shuffle(pairs)
        flat_indices = [p[0] for p in pairs]
        return flat_indices, by_cm

    def _pick_four_flat_for_quad(self, pool, rng):
        """从该类本次抽中的索引列表里选 4 个用于 2×2；尽量互不重复。"""
        if not pool:
            return []
        uniq = list(dict.fromkeys(pool))
        if len(uniq) >= 4:
            return rng.sample(uniq, 4)
        out = list(uniq)
        while len(out) < 4:
            out.append(rng.choice(pool))
        return out[:4]

    def _build_cm_quad_montages(self, by_cm, out_dir, cell, resample, seed):
        """从各类在本次 200 张中的子集里，用独立可复现 RNG(seed+1) 各抽 4 张拼成 2×2，输出四张 PNG。"""
        if getattr(self.hparams, 'no_error_cm_quad', False):
            return
        rng_q = random.Random(int(seed) + 1)
        dataset_test = self.trainer.datamodule.dataset_test
        order = ('TP', 'FP', 'FN', 'TN')
        prefix = getattr(self.hparams, 'error_cm_quad_filename_prefix', 'cm_quad')
        w = h = int(cell)
        for cm in order:
            picks = self._pick_four_flat_for_quad(by_cm.get(cm, []), rng_q)
            if len(picks) < 4:
                continue
            canvas = Image.new('RGB', (2 * w, 2 * h), (255, 255, 255))
            for k, fi in enumerate(picks):
                r, c = divmod(k, 2)
                pth = dataset_test.samples[fi][0]
                try:
                    im = Image.open(pth).convert('RGB').resize((w, h), resample)
                except OSError:
                    im = Image.new('RGB', (w, h), (220, 220, 220))
                canvas.paste(im, (c * w, r * h))
            out_path = os.path.join(out_dir, f'{prefix}_{cm}.png')
            canvas.save(out_path, 'PNG', optimize=True)
            print(f'[error analysis] CM 2x2 quad ({cm}) saved to {out_path}')

    def _build_cm_montage_1000(self):
        """测试结束后：从 TP/FP/FN/TN 分层随机抽原图，拼成 error_cm_montage_rows×cols 大图（默认 20×10=200）。"""
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        if not getattr(self.hparams, 'save_error_images', False):
            return
        if getattr(self.hparams, 'no_error_cm_montage', False):
            return
        total = sum(len(self._flat_by_cm[k]) for k in ('TP', 'FP', 'FN', 'TN'))
        if total == 0:
            return
        rows = int(getattr(self.hparams, 'error_cm_montage_rows', 20))
        cols = int(getattr(self.hparams, 'error_cm_montage_cols', 10))
        n_total = rows * cols
        if n_total <= 0:
            rows, cols, n_total = 20, 10, 200
        cell = int(getattr(self.hparams, 'error_cm_montage_cell_size', 128))
        seed = int(getattr(self.hparams, 'error_cm_montage_seed', 42))
        out_name = getattr(self.hparams, 'error_cm_montage_filename', 'cm_montage_20x10.png')
        rng = random.Random(seed)
        flat_indices, by_cm = self._sample_flat_indices_cm_montage(rng, n_total)
        dataset_test = self.trainer.datamodule.dataset_test
        paths = [dataset_test.samples[i][0] for i in flat_indices]
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        w = h = cell
        canvas = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
        for idx, pth in enumerate(paths[: rows * cols]):
            r, c = divmod(idx, cols)
            try:
                im = Image.open(pth).convert('RGB').resize((w, h), resample)
            except OSError:
                im = Image.new('RGB', (w, h), (220, 220, 220))
            canvas.paste(im, (c * w, r * h))
        out_dir = self.hparams.error_output_dir
        out_path = os.path.join(out_dir, out_name)
        canvas.save(out_path, 'PNG', optimize=True)
        print(f'[error analysis] CM montage ({rows}x{cols}) saved to {out_path}')
        self._build_cm_quad_montages(by_cm, out_dir, cell, resample, seed)

    def _tensor_chw_to_rgb_uint8(self, t_chw: torch.Tensor) -> np.ndarray:
        """反 ImageNet 归一化，得到 HWC uint8 RGB。"""
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=t_chw.dtype, device=t_chw.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=t_chw.dtype, device=t_chw.device).view(3, 1, 1)
        x = (t_chw * std + mean).clamp(0.0, 1.0)
        hwc = (x * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        return np.asarray(hwc)

    def _save_pil_png_pdf(self, img: Image.Image, out_dir: str, out_base: str):
        out_png = os.path.join(out_dir, f'{out_base}.png')
        out_pdf = os.path.join(out_dir, f'{out_base}.pdf')
        img.save(out_png, 'PNG', optimize=True)
        img.save(out_pdf, 'PDF', resolution=300.0)
        return out_png, out_pdf

    def _build_fn_original_vs_input_montage(self, out_dir: str) -> None:
        """FN 展示图：原图与模型输入分别保存。"""
        if getattr(self.hparams, 'no_error_fn_montage', False):
            return
        rows = int(getattr(self.hparams, 'error_fn_montage_rows', 7))
        half_cols = int(getattr(self.hparams, 'error_fn_montage_half_cols', 5))
        n_target = rows * half_cols
        cell = int(getattr(self.hparams, 'error_fn_montage_cell_size', 128))
        out_base = getattr(self.hparams, 'error_fn_montage_basename', 'fn_original_vs_input_montage')
        out_base_orig = getattr(self.hparams, 'error_fn_original_montage_basename', f'{out_base}_original')
        out_base_input = getattr(self.hparams, 'error_fn_input_montage_basename', f'{out_base}_input')
        rows_data = list(getattr(self, '_fn_showcase_rows', []))
        if not rows_data:
            return
        while len(rows_data) < n_target:
            rows_data.append((None, None))
        rows_data = rows_data[:n_target]
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        w = h = cell

        # --- 1) 原始图拼图：锁定纵横比缩放到单元格内（不拉伸），居中留白 ---
        canvas_orig = Image.new('RGB', (half_cols * w, rows * h), (255, 255, 255))
        for idx in range(n_target):
            r, c = divmod(idx, half_cols)
            src_path, _ = rows_data[idx]
            if src_path and os.path.isfile(src_path):
                try:
                    im_l = Image.open(src_path).convert('RGB')
                    im_fit = im_l.copy()
                    im_fit.thumbnail((w, h), resample)
                except OSError:
                    im_fit = Image.new('RGB', (w, h), (220, 220, 220))
            else:
                im_fit = Image.new('RGB', (w, h), (235, 235, 235))
            pw = c * w + max(0, (w - im_fit.size[0]) // 2)
            ph = r * h + max(0, (h - im_fit.size[1]) // 2)
            canvas_orig.paste(im_fit, (pw, ph))
        out_png_o, out_pdf_o = self._save_pil_png_pdf(canvas_orig, out_dir, out_base_orig)

        # --- 2) 模型输入拼图：统一网格 ---
        canvas_input = Image.new('RGB', (half_cols * w, rows * h), (255, 255, 255))
        for idx in range(n_target):
            r, c = divmod(idx, half_cols)
            _, t_chw = rows_data[idx]
            if t_chw is not None and t_chw.numel() > 0:
                try:
                    arr = self._tensor_chw_to_rgb_uint8(t_chw)
                    im_r = Image.fromarray(arr).resize((w, h), resample)
                except Exception:
                    im_r = Image.new('RGB', (w, h), (220, 220, 220))
            else:
                im_r = Image.new('RGB', (w, h), (235, 235, 235))
            canvas_input.paste(im_r, (c * w, r * h))
        out_png_i, out_pdf_i = self._save_pil_png_pdf(canvas_input, out_dir, out_base_input)

        print(
            f'[error analysis] FN montages saved: '
            f'original={out_png_o} / {out_pdf_o}, input={out_png_i} / {out_pdf_i}'
        )

    def _plot_confusion_scatter_unified(self, out_path):
        """Single axes: x=index, y=P(fire). Legend: TP, FP, FN, TN.

        Colors #83D6BC / #FEA492: TP/FN (coral) vs FP/TN (mint); circles vs crosses unchanged.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        rc = {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 15,
            'axes.labelsize': 16,
            'axes.titlesize': 17,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'axes.unicode_minus': False,
            'axes.linewidth': 0.9,
            'axes.edgecolor': '#2C2C2C',
            'axes.labelcolor': '#1A1A1A',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'grid.color': '#B8B8B8',
            'grid.linestyle': '--',
            'grid.linewidth': 0.45,
        }
        # Plot & legend order: TP → FP → FN → TN (top to bottom in default legend)
        order = ('TP', 'FP', 'FN', 'TN')
        col_mint, col_coral = '#83D6BC', '#FEA492'
        edge_mint, edge_coral = '#4A9B8A', '#D4735A'
        with plt.rc_context(rc=rc):
            fig, ax = plt.subplots(figsize=(10, 6.8))
            all_idx, all_s = [], []
            for k in order:
                for ix, sc in self._scatter_by_cm[k]:
                    all_idx.append(ix)
                    all_s.append(sc)
            if all_idx:
                pad_x = max(1, int((max(all_idx) - min(all_idx)) * 0.02) + 1)
                pad_y = 0.02
                xlim = (min(all_idx) - pad_x, max(all_idx) + pad_x)
                ylim = (max(0.0, min(all_s) - pad_y), min(1.0, max(all_s) + pad_y))
            else:
                xlim, ylim = (0, 1), (0, 1)

            legend_handles = []
            legend_labels = []
            for key in order:
                pts = self._scatter_by_cm[key]
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if key in ('TP', 'TN'):
                    h = ax.scatter(
                        xs, ys, s=13, alpha=0.88, zorder=3,
                        c=col_coral if key == 'TP' else col_mint,
                        marker='o',
                        edgecolors=edge_coral if key == 'TP' else edge_mint,
                        linewidths=0.55,
                    )
                else:
                    h = ax.scatter(
                        xs, ys, s=26, alpha=0.92, zorder=3,
                        c=col_coral if key == 'FN' else col_mint,
                        marker='x',
                        linewidths=1.15,
                    )
                legend_handles.append(h)
                legend_labels.append(key)
            ax.axhline(0.5, color='#555555', linestyle='-', linewidth=0.9, alpha=0.85, zorder=2)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, which='major', axis='both')
            ax.set_axisbelow(True)
            off = int(getattr(self.hparams, 'error_neg_index_offset', 2500))
            ax.set_xlabel(f'Image index')
            ax.set_ylabel('Fire Prediction Score')
            if legend_handles:
                ax.legend(
                    legend_handles, legend_labels,
                    ncol=4,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 1.02),
                    framealpha=0.92,
                    edgecolor='#CCCCCC',
                )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            save_figure_png_and_pdf(
                fig, out_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none'
            )
            plt.close(fig)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.hparams.multi_scale is not None:
            if self.current_epoch == int(self.hparams.multi_scale.split('_')[1]):
                # image_size = self.hparams.image_size
                self.trainer.reset_train_dataloader(self)

    def _update_metrics(self, preds, labels, mode="val"):
        """更新指标"""
        if mode == "val":
            self.val_accuracy(preds, labels)
            self.val_precision(preds, labels)
            self.val_recall(preds, labels)
            self.val_f1(preds, labels)
            self.val_precision_per_class(preds, labels)
            self.val_recall_per_class(preds, labels)
            self.val_f1_per_class(preds, labels)
        else:  # test
            self.test_accuracy(preds, labels)
            self.test_precision(preds, labels)
            self.test_recall(preds, labels)
            self.test_f1(preds, labels)
            self.test_precision_per_class(preds, labels)
            self.test_recall_per_class(preds, labels)
            self.test_f1_per_class(preds, labels)

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        if mode == "train" and self.mixup_fn is not None:
            imgs, labels = self.mixup_fn(imgs, labels)
        aux_losses = None
        use_paper_loss = bool(getattr(self.hparams, 'use_mgm_fes_loss', False))
        if mode == "train" and use_paper_loss and hasattr(self.model, "forward_with_aux"):
            preds, aux_losses = self.model.forward_with_aux(imgs)
        else:
            preds = self.model(imgs)

        if mode == "train":
            if self.distillation_type == 'none':
                final_loss = self.criterion(preds, labels)
            else:
                final_loss = self.criterion(imgs, preds, labels)

            if use_paper_loss and aux_losses is not None:
                w_final = float(getattr(self.hparams, 'mgm_fes_loss_final_weight', 1.0))
                w_back = float(getattr(self.hparams, 'mgm_fes_loss_back_weight', 0.0))
                w_joint = float(getattr(self.hparams, 'mgm_fes_loss_joint_weight', 0.0))
                w_drop = float(getattr(self.hparams, 'mgm_fes_loss_drop_weight', 0.0))
                loss = (
                    w_final * final_loss
                    + w_back * aux_losses["loss_back"]
                    + w_joint * aux_losses["loss_joint"]
                    + w_drop * aux_losses["loss_drop"]
                )
                self.log("train_loss_final", final_loss)
                self.log("train_loss_back", aux_losses["loss_back"])
                self.log("train_loss_joint", aux_losses["loss_joint"])
                self.log("train_loss_drop", aux_losses["loss_drop"])
            else:
                loss = final_loss
            self.log("%s_loss" % mode, loss)
        else:
            loss = self.criterion_eva(preds, labels)
            acc1, acc5 = accuracy(preds, labels, topk=(1, 5))
            
            # 更新指标
            self._update_metrics(preds, labels, mode)
            
            # 记录基本指标
            self.log("%s_loss" % mode, loss)
            self.log("%s_acc1" % mode, acc1)
            self.log("%s_acc5" % mode, acc5)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def on_test_epoch_start(self):
        if not getattr(self.hparams, 'save_error_images', False):
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        self._idx_pos = 0
        self._idx_neg = 0
        self._all_inf_lines = []
        self._scatter_by_cm = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
        self._flat_by_cm = {'TP': [], 'FP': [], 'FN': [], 'TN': []}
        self._test_flat_idx = 0
        self._fp_dist_paths = []
        self._fn_dist_paths = []
        self._fn_showcase_rows = []
        os.makedirs(self.hparams.error_output_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion_eva(preds, labels)
        acc1, acc5 = accuracy(preds, labels, topk=(1, 5))
        self._update_metrics(preds, labels, mode="test")
        self.log("test_loss", loss)
        self.log("test_acc1", acc1)
        self.log("test_acc5", acc5)
        if getattr(self.hparams, 'save_error_images', False):
            self._save_error_analysis_samples(imgs.detach(), labels, preds.detach())
    
    def on_validation_epoch_end(self):
        """验证epoch结束时计算和记录指标"""
        # 计算并记录验证指标
        self.log("val_accuracy", self.val_accuracy.compute())
        self.log("val_precision_macro", self.val_precision.compute())
        self.log("val_recall_macro", self.val_recall.compute())
        self.log("val_f1_macro", self.val_f1.compute())
        
        # 记录每个类别的指标
        precision_per_class = self.val_precision_per_class.compute()
        recall_per_class = self.val_recall_per_class.compute()
        f1_per_class = self.val_f1_per_class.compute()
        
        for i in range(len(precision_per_class)):
            self.log(f"val_precision_class_{i}", precision_per_class[i])
            self.log(f"val_recall_class_{i}", recall_per_class[i])
            self.log(f"val_f1_class_{i}", f1_per_class[i])
        
        # 重置指标
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_precision_per_class.reset()
        self.val_recall_per_class.reset()
        self.val_f1_per_class.reset()
    
    def on_test_epoch_end(self):
        """测试epoch结束时计算和记录指标"""
        if getattr(self.hparams, 'save_error_images', False) and self.trainer.is_global_zero:
            out_dir = self.hparams.error_output_dir
            err_path = os.path.join(out_dir, 'error_inf.txt')
            with open(err_path, 'w', encoding='utf-8') as f:
                for name in self._all_inf_lines:
                    f.write(name + '\n')
            scatter_path = os.path.join(out_dir, 'confusion_scatter.png')
            self._plot_confusion_scatter_unified(scatter_path)
            self._export_sample_far_from_half()
            self._build_cm_montage_1000()
            self._build_fn_original_vs_input_montage(out_dir)
        # 计算并记录测试指标
        self.log("test_accuracy", self.test_accuracy.compute())
        self.log("test_precision_macro", self.test_precision.compute())
        self.log("test_recall_macro", self.test_recall.compute())
        self.log("test_f1_macro", self.test_f1.compute())
        
        # 记录每个类别的指标
        precision_per_class = self.test_precision_per_class.compute()
        recall_per_class = self.test_recall_per_class.compute()
        f1_per_class = self.test_f1_per_class.compute()
        
        for i in range(len(precision_per_class)):
            self.log(f"test_precision_class_{i}", precision_per_class[i])
            self.log(f"test_recall_class_{i}", recall_per_class[i])
            self.log(f"test_f1_class_{i}", f1_per_class[i])
        
        # 重置指标
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_precision_per_class.reset()
        self.test_recall_per_class.reset()
        self.test_f1_per_class.reset()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.parameters())
        if self.hparams.sched == 'cosine':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                            warmup_epochs=self.hparams.warmup_epochs,
                            max_epochs=self.hparams.epochs,
                            warmup_start_lr=self.hparams.warmup_lr,
                            eta_min=self.hparams.min_lr
                        )
        else:
            # scheduler, _ = create_scheduler(self.hparams, optimizer)
            raise NotImplementedError

        return [optimizer], [scheduler]


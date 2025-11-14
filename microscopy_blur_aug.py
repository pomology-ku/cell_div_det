# microscopy_blur_aug.py
# Albumentations を使った「ブラー耐性」向けの学習用Aug（提案A）

from typing import List
import albumentations as A

def build_train_aug() -> List[A.BasicTransform]:
    """
    UltralyticsのPython API `augmentations=` にそのまま渡せる「リスト」を返す。
    - ここを自由に編集してブラー分布を調整してください。
    """
    blur_block = A.OneOf([
        A.Defocus(radius=(2, 8), alias_blur=(0.05, 0.4), p=0.35),     # 焦点ズレ
        A.MotionBlur(blur_limit=(7, 21), p=0.35),                      # 動体/走査ブレ
        A.GlassBlur(sigma=0.5, max_delta=3, iterations=2, p=0.15),     # 非線形ブレ
        A.Downscale(scale_min=0.25, scale_max=0.6, p=0.15),            # 低周波化（縮小→拡大）
    ], p=0.8)

    photometric = A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=1.0),
    ], p=0.5)

    # Ultralytics 側が内部で Compose を構築するため、
    # ここでは「適用したいTransformを配列で返す」だけでOK
    return [
        # blur_block,
        # photometric,
        # A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        # A.ImageCompression(quality_lower=50, quality_upper=95, p=0.2),
        A.Defocus(radius=(2, 8), alias_blur=(0.5,1), p=0.5)
    ]

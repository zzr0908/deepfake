from PIL import Image
import os
import shutil
import pandas as pd
from tqdm import tqdm


def is_image_invalid(image_path):
    try:
        img = Image.open(image_path)
        img.load()
        return False
    except Exception as E:
        return True


if __name__ == '__main__':
    df_train_label = pd.read_csv("phase1/trainset_label.txt", sep=',')
    df_val_label = pd.read_csv("phase1/valset_label.txt", sep=',')

    for d in ["phase1_images",
              "phase1_images/train",
              "phase1_images/val",
              "phase1_images/train/1",
              "phase1_images/train/0",
              "phase1_images/val/1",
              "phase1_images/val/0"
              ]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    for idx, row in tqdm(df_train_label.iterrows()):
        if not os.path.isfile(os.path.join("phase1/trainset/", row.img_name)) or is_image_invalid(
                f"phase1/trainset/{row.img_name}"):
            continue
        source_file = f"phase1/trainset/{row.img_name}"
        target_file = f"phase1_images/train/{row.target}/{row.img_name}"
        shutil.copy2(source_file, target_file)

    for idx, row in tqdm(df_val_label.iterrows()):
        if not os.path.isfile(
                os.path.join("phase1/valset/", row.img_name) or is_image_invalid(f"phase1/valset/{row.img_name}")):
            continue
        source_file = f"phase1/valset/{row.img_name}"
        target_file = f"phase1_images/val/{row.target}/{row.img_name}"
        shutil.copy2(source_file, target_file)
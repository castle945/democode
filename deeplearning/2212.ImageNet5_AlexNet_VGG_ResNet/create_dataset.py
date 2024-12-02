import random
from pathlib import Path
import os
from tqdm import tqdm

split_rate = 0.1

# ori_data_root = "/datasets/flower_photos/"
# classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# new_data_root = "/datasets/flower5/"

ori_data_root = "/datasets/imagenet2012/train/"
classes = ['n01440764', 'n01514668', 'n01518878', 'n01560419', 'n01582220'] # 鱼、鸡、鸵鸟、灰鸟、黑鸟
new_data_root = "/datasets/imagenet5/"

def main():
    random.seed(0)

    Path(new_data_root).mkdir(exist_ok=True)
    train_root = Path(new_data_root + "train")
    train_root.mkdir(exist_ok=True)
    val_root = Path(new_data_root + "val")
    val_root.mkdir(exist_ok=True)

    for cls in tqdm(classes):
        train_dir = Path(os.path.join(train_root, cls))
        train_dir.mkdir(exist_ok=True)
        val_dir = Path(os.path.join(val_root, cls))
        val_dir.mkdir(exist_ok=True)

        cls_dir = os.path.join(ori_data_root, cls)
        filenames = os.listdir(cls_dir)
        eval_filenames = random.sample(filenames, k=int(len(filenames)*split_rate))
        for filename in filenames:
            filepath = os.path.join(cls_dir, filename)
            if filename in eval_filenames:
                os.system(f"cp {filepath} {val_dir}")
            else:
                os.system(f"cp {filepath} {train_dir}")

if __name__ == '__main__':
    main()
import os
import shutil
import random
import cv2

SRC = "datasets/Iranian Plate.v1"
DST = "datasets/Iranian Plate.plate1"
EXTS = [".jpg", ".jpeg", ".png"]

def find_image(folder, base):
    for e in EXTS:
        p = os.path.join(folder, base + e)
        if os.path.exists(p):
            return p, e
    return None, None

def main():
    random.seed(42)
    shutil.rmtree(DST, ignore_errors=True)

    for split in ["train", "valid", "test"]:
        src_lab = os.path.join(SRC, split, "labels")
        dst_img = os.path.join(DST, split, "images")
        dst_lab = os.path.join(DST, split, "labels")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lab, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(src_lab)):
            if not fn.endswith(".txt"):
                continue
            base = fn[:-4]
            src_img_path, ext = find_image(os.path.join(SRC, split, "images"), base)
            if src_img_path is None:
                continue

            # ادغام کلاس‌ها: همه -> 0 (plate)
            with open(os.path.join(src_lab, fn), encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if len(l.split()) >= 5]
            dst_label = os.path.join(dst_lab, fn)
            with open(dst_label, "w", encoding="utf-8") as f:
                for l in lines:
                    p = l.split()
                    f.write(" ".join(["0"] + p[1:]) + "\n")
            shutil.copy(src_img_path, os.path.join(dst_img, base + ext))
            n += 1

            # augmentation آفلاین فقط برای train (blur و نور؛ بدون جابه‌جایی باکس)
            if split == "train":
                img = cv2.imread(src_img_path)
                if img is None:
                    continue
                if random.random() < 0.3:
                    b = cv2.GaussianBlur(img, (5, 5), 0)
                    cv2.imwrite(os.path.join(dst_img, base + "_blur" + ext), b)
                    shutil.copy(dst_label, os.path.join(dst_lab, base + "_blur.txt"))
                if random.random() < 0.3:
                    alpha = random.uniform(0.7, 1.3)
                    beta = random.randint(-30, 30)
                    l = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    cv2.imwrite(os.path.join(dst_img, base + "_light" + ext), l)
                    shutil.copy(dst_label, os.path.join(dst_lab, base + "_light.txt"))
        print(f"✅ {split}: {n} تصویر آماده شد")

    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['plate']\n")
    print(f"💾 دیتاست تک‌کلاسی ساخته شد: {DST}")

if __name__ == "__main__":
    main()
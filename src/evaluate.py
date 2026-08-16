import os
import sys
import glob
import random
import json
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detect_pipeline import IranianLPRPipeline
from ocr_engine import read_iranian_plate


NUM_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
PLATE_CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
MODEL_PATH = sys.argv[4] if len(sys.argv) > 4 else None  # 👈 این خط رو داری؟

CONF_TAG = f"{PLATE_CONF:.2f}".replace(".", "_")
MODEL_TAG = os.path.splitext(os.path.basename(MODEL_PATH))[0] if MODEL_PATH else "plate_detector"
OUT_PATH = f"outputs/eval_report_{SPLIT}_{MODEL_TAG}_conf{CONF_TAG}.json"

IOU_THR = 0.3
CONF_TAG = f"{PLATE_CONF:.2f}".replace(".", "_")
TEST_DIR = f"datasets/Iranian Plate.v1/{SPLIT}"
ERR_DIR = "outputs/evaluation_errors"

TAG_COLORS = {
    "TP-GT": (0, 255, 0),
    "FN-V":  (0, 0, 255),
    "FN-P":  (0, 165, 255),
    "TP-P":  (255, 0, 0),
    "FP-U":  (0, 255, 255),
    "FP-H":  (255, 0, 255),
}

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(1, a1 + a2 - inter)

def center(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
def inside(b, pt): return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]

def read_all_gt(label_path, w, h):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5: continue
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append([int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                              int((cx + bw / 2) * w), int((cy + bh / 2) * h)])
    return boxes

def match_one_to_one(gt_boxes, pred_boxes, thr=IOU_THR):
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            v = iou(g, p)
            if v > thr:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    used_g, used_p = set(), set()
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
    unmatched_g = [i for i in range(len(gt_boxes)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred_boxes)) if i not in used_p]
    return len(used_g), len(unmatched_p), len(unmatched_g), unmatched_g, unmatched_p

def save_error_image(img, base, gt_boxes, gt_tags, pred_boxes, pred_tags):
    vis = img.copy()
    for g, tag in zip(gt_boxes, gt_tags):
        c = TAG_COLORS[tag]
        cv2.rectangle(vis, (g[0], g[1]), (g[2], g[3]), c, 2)
        cv2.putText(vis, tag, (g[0], max(g[1] - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    for p, tag in zip(pred_boxes, pred_tags):
        c = TAG_COLORS[tag]
        cv2.rectangle(vis, (p[0], p[1]), (p[2], p[3]), c, 2)
        cv2.putText(vis, tag, (p[0], min(p[3] + 14, img.shape[0] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    os.makedirs(ERR_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(ERR_DIR, f"{SPLIT}_conf{CONF_TAG}_{base}.jpg"), vis)

def main():
    
    pipeline = IranianLPRPipeline(plate_model_path=MODEL_PATH)

    images = sorted(glob.glob(os.path.join(TEST_DIR, "images", "*.jpg")) +
                    glob.glob(os.path.join(TEST_DIR, "images", "*.png")))
    random.seed(42)
    random.shuffle(images)
    images = images[:NUM_IMAGES]
    print(f"📊 ارزیابی روی {len(images)} تصویر از پوشه {SPLIT} (IoU>{IOU_THR} | plate conf>{PLATE_CONF})...")

    stats = dict(gt=0, pred_vehicles=0, pred_plates=0, tp=0, fp=0, fn=0, ocr_out=0,
                 fn_vehicle=0, fn_plate=0, fp_unann=0, fp_hall=0)
    conf_v, conf_p = [], []
    missed_images, per_image = [], []
    saved_errors = 0

    for idx, path in enumerate(images, 1):
        img = cv2.imread(path)
        if img is None: continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(path))[0]

        gt_boxes = read_all_gt(os.path.join(TEST_DIR, "labels", base + ".txt"), w, h)
        stats["gt"] += len(gt_boxes)

        vehicles = pipeline.detect_vehicles(img)
        vehicle_boxes = [v["bbox"] for v in vehicles]
        stats["pred_vehicles"] += len(vehicles)
        conf_v += [v["confidence"] for v in vehicles]

        det_plate_boxes, pred_parents, ocr_count = [], [], 0
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            plate = pipeline.detect_plate(img[vy1:vy2, vx1:vx2], conf=PLATE_CONF)
            if plate is None:
                continue
            stats["pred_plates"] += 1
            conf_p.append(plate["confidence"])
            px1, py1, px2, py2 = plate["bbox"]
            det_plate_boxes.append([vx1+px1, vy1+py1, vx1+px2, vy1+py2])
            pred_parents.append(v["bbox"])

            try:
                text = read_iranian_plate(img[vy1+py1:vy1+py2, vx1+px1:vx1+px2])
                if text and text != "خوانده نشد":
                    ocr_count += 1
                    stats["ocr_out"] += 1
            except Exception:
                pass

        tp, fp, fn, unmatched_g, unmatched_p = match_one_to_one(gt_boxes, det_plate_boxes)
        stats["tp"] += tp; stats["fp"] += fp; stats["fn"] += fn

        gt_tags = []
        for gi, g in enumerate(gt_boxes):
            if gi not in unmatched_g:
                gt_tags.append("TP-GT")
            elif any(inside(vb, center(g)) for vb in vehicle_boxes):
                gt_tags.append("FN-P"); stats["fn_plate"] += 1
            else:
                gt_tags.append("FN-V"); stats["fn_vehicle"] += 1

        pred_tags = []
        for pi in range(len(det_plate_boxes)):
            if pi not in unmatched_p:
                pred_tags.append("TP-P")
            elif any(inside(pred_parents[pi], center(g)) for g in gt_boxes):
                pred_tags.append("FP-H"); stats["fp_hall"] += 1
            else:
                pred_tags.append("FP-U"); stats["fp_unann"] += 1

        if gt_boxes and not det_plate_boxes:
            missed_images.append(base)

        if fn or fp:
            save_error_image(img, base, gt_boxes, gt_tags, det_plate_boxes, pred_tags)
            saved_errors += 1

        print(f"[{idx}/{len(images)}] {base[:25]:25s} | GT:{len(gt_boxes)} | P:{len(det_plate_boxes)} | TP:{tp} FP:{fp} FN:{fn} | OCR:{ocr_count}")
        per_image.append({"file": base, "gt": len(gt_boxes), "pred": len(det_plate_boxes),
                          "tp": tp, "fp": fp, "fn": fn, "ocr_out": ocr_count,
                          "fn_vehicle": gt_tags.count("FN-V"), "fn_plate": gt_tags.count("FN-P"),
                          "fp_unann": pred_tags.count("FP-U"), "fp_hall": pred_tags.count("FP-H")})

    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    ocr_rate = stats["ocr_out"] / stats["pred_plates"] if stats["pred_plates"] else 0
    real_precision = tp / (tp + stats["fp_hall"]) if (tp + stats["fp_hall"]) else 0

    print("\n" + "=" * 60)
    print(f"📊 خلاصه ارزیابی + تفکیک خطا (پوشه: {SPLIT} | plate conf>{PLATE_CONF})")
    print("=" * 60)
    print(f"تصاویر بررسی‌شده:        {len(images)}")
    print(f"TP / FP / FN:            {tp} / {fp} / {fn}")
    print(f"Precision:               {precision:.1%}   |   Precision واقعی (بدون FP-U): {real_precision:.1%}")
    print(f"Recall:                  {recall:.1%}")
    print(f"F1:                      {f1:.1%}")
    print(f"نرخ خروجی OCR:           {stats['ocr_out']} از {stats['pred_plates']} ({ocr_rate:.1%})")
    print("-" * 60)
    print(f"🔍 FN تفکیک ({fn}):       {stats['fn_vehicle']} خودرو پیدا نشد / {stats['fn_plate']} پلاک پیدا نشد")
    print(f"🔍 FP تفکیک ({fp}):       {stats['fp_unann']} پلاک واقعی بدون لیبل / {stats['fp_hall']} تشخیص غلط")
    print(f"🖼️ تصاویر خطا:            {saved_errors} تا در {ERR_DIR}")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"split": SPLIT, "plate_conf": PLATE_CONF, "iou_threshold": IOU_THR,
                   "model": MODEL_TAG, "summary": stats,  # 👈 این خط
                   "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                   "missed": missed_images, "per_image": per_image},
                  f, ensure_ascii=False, indent=2)
    print(f"💾 گزارش کامل: {OUT_PATH}")

if __name__ == "__main__":
    main()
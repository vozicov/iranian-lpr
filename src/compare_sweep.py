import glob
import json

def main():
    files = sorted(glob.glob("outputs/eval_report_*_conf*.json"))
    if not files:
        print("❌ گزارشی پیدا نشد.")
        return
    print(f"{'conf':>5} | {'model':>18} | {'split':>5} | {'TP':>4} {'FP':>4} {'FN':>4} | {'Prec':>7} {'Rec':>7} {'F1':>7} | {'FN-V':>4} {'FN-P':>4} {'FP-U':>4} {'FP-H':>4}")
    print("-" * 115)
    for p in files:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        s = d["summary"]
        print(f"{d['plate_conf']:>5} | {d.get('model', 'plate_detector'):>18} | {d['split']:>5} | {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} | "
              f"{d['precision']:>7.1%} {d['recall']:>7.1%} {d['f1']:>7.1%} | "
              f"{s['fn_vehicle']:>4} {s['fn_plate']:>4} {s['fp_unann']:>4} {s['fp_hall']:>4}")

if __name__ == "__main__":
    main()  
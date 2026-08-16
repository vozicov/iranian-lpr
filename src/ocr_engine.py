import cv2
import os
import re
import tempfile

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
PLATE_LETTERS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
ALLOWLIST = FA_DIGITS + PLATE_LETTERS + "0123456789"

_engine = None
_engine_kind = None

def get_ocr_engine():
    """اول مدل تخصصی CRNN پلاک فارسی؛ اگه نشد، EasyOCR"""
    global _engine, _engine_kind
    if _engine is not None:
        return _engine, _engine_kind

    try:
        from hezar.models import Model
        print("⏳ بارگذاری مدل تخصصی OCR پلاک فارسی (CRNN)...")
        _engine = Model.load("hezarai/crnn-fa-license-plate-recognition-v2")
        _engine_kind = "crnn"
        print("✅ مدل تخصصی CRNN بارگذاری شد!")
    except Exception as e:
        print(f"⚠️ CRNN در دسترس نیست ({e})؛ استفاده از EasyOCR")
        import easyocr
        _engine = easyocr.Reader(['fa'], gpu=False, verbose=False)
        _engine_kind = "easyocr"
    return _engine, _engine_kind

def preprocess_plate(plate_img):
    """بزرگ‌نمایی + حاشیه سفید"""
    h, w = plate_img.shape[:2]
    up = cv2.resize(plate_img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    up = cv2.copyMakeBorder(up, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return up

def normalize_text(text):
    map_ch = {"ي": "ی", "ك": "ک", "ة": "ه", "‌": "", ".": "", ",": "", "|": ""}
    out = []
    for ch in text:
        if ch in "0123456789":
            out.append(FA_DIGITS[int(ch)])
        elif ch in "٠١٢٣٤٥٦٧٨٩":
            out.append(FA_DIGITS[int(ch)])
        elif ch in map_ch:
            out.append(map_ch[ch])
        else:
            out.append(ch)
    return "".join(out)

def structure_plate(text):
    clean = "".join(text.split())
    clean = re.sub(r"[{}'\":]", "", clean)
    clean = re.sub(r"[a-zA-Z]", "", clean)

    # حالت ۱ (چپ‌به‌راست): [کد][۳رقم][حرف][۲رقم]
    m = re.search(r"([۰-۹]{2})([۰-۹]{3})([^۰-۹])([۰-۹]{2})", clean)
    if m:
        code, three, letter, two = m.groups()
        return f"{code} ایران - {three} {letter} {two}"

    # حالت ۲ (خروجی CRNN راست‌به‌چپ): [۲رقم][حرف][۳رقم][کد]
    m = re.search(r"([۰-۹]{2})([^۰-۹])([۰-۹]{3})([۰-۹]{2})", clean)
    if m:
        two, letter, three, code = m.groups()
        return f"{code} ایران - {three} {letter} {two}"

    return clean if clean else "خوانده نشد"

def _extract_text(out):
    if isinstance(out, dict):                   # 👈 خروجی مدل CRNN دیکشنریه
        return str(out.get("text", ""))
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        parts = []
        for x in out:
            if isinstance(x, dict):
                parts.append(str(x.get("text", x.get("label", ""))))
            else:
                parts.append(str(x))
        return " ".join(parts)
    return str(out)

def read_iranian_plate(plate_img):
    engine, kind = get_ocr_engine()
    proc = preprocess_plate(plate_img)

    if kind == "crnn":
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name
            cv2.imwrite(tmp_path, proc)
            out = engine.predict(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        text = normalize_text(_extract_text(out))
    else:
        res = engine.readtext(proc, detail=0, paragraph=True, allowlist=ALLOWLIST)
        text = normalize_text(" ".join(res))

    return structure_plate(text)
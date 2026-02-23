"""
Extract on-screen timestamp from video frames (top-left region).
Used when video is a clip and doesn't start at journey start - match on-screen time to Excel/RTIS data.
"""
import re
from datetime import datetime, date, time
from typing import Optional, Union
import cv2
import numpy as np

_ocr_available = False
_ocr_reader = None
_ocr_backend = ''  # 'easyocr' or 'pytesseract'
_ocr_initialized = False

def _init_ocr():
    """Lazy init OCR - try EasyOCR first, fallback to pytesseract."""
    global _ocr_available, _ocr_reader, _ocr_backend, _ocr_initialized
    if _ocr_initialized:
        return _ocr_available
    _ocr_initialized = True
    try:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        _ocr_backend = 'easyocr'
        _ocr_available = True
        return True
    except Exception:
        pass
    try:
        import pytesseract
        _ocr_reader = pytesseract
        _ocr_backend = 'pytesseract'
        _ocr_available = True
        return True
    except Exception:
        pass
    _ocr_backend = ''
    _ocr_available = False
    return False


def _parse_time_from_text(text: str) -> Optional[time]:
    """Parse HH:MM:SS or HH:MM from OCR text."""
    if not text or not isinstance(text, str):
        return None
    # HH:MM:SS
    m = re.search(r'\b(\d{1,2}):(\d{2}):(\d{2})\b', text)
    if m:
        try:
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= s <= 59:
                return time(h, mi, s)
        except (ValueError, IndexError):
            pass
    # HH:MM
    m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
    if m:
        try:
            h, mi = int(m.group(1)), int(m.group(2))
            if 0 <= h <= 23 and 0 <= mi <= 59:
                return time(h, mi, 0)
        except (ValueError, IndexError):
            pass
    return None


def extract_time_from_frame(frame: np.ndarray, reference_date: Optional[Union[date, datetime]] = None) -> Optional[datetime]:
    """
    Extract on-screen time from top-left region of frame.
    Args:
        frame: BGR image (OpenCV format)
        reference_date: Date to use for returned datetime (from RTIS/Excel). If None, uses today.
    Returns:
        datetime or None if OCR failed
    """
    if frame is None or frame.size == 0:
        return None
    if not _init_ocr():
        return None

    h, w = frame.shape[:2]
    # Crop top-left: ~25% width, ~15% height (where clock usually appears)
    x1, y1 = 0, 0
    x2 = int(0.25 * w)
    y2 = int(0.15 * h)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Preprocess for better OCR: grayscale, contrast
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = ''
    try:
        if _ocr_backend == 'pytesseract':
            text = _ocr_reader.image_to_string(gray, config='--psm 7 digits')
        elif _ocr_backend == 'easyocr':
            results = _ocr_reader.readtext(gray, detail=0)
            text = ' '.join(str(r) for r in results) if results else ''
        else:
            return None
    except Exception:
        return None

    t = _parse_time_from_text(text)
    if t is None:
        return None

    ref = reference_date
    if ref is None:
        ref = date.today()
    elif isinstance(ref, datetime):
        ref = ref.date()
    return datetime.combine(ref, t)


def is_ocr_available() -> bool:
    """Check if OCR is available."""
    return _init_ocr()

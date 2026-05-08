import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ══════════════════════════════════════════════
COIN_CLASS_NAME       = "Coin"
COIN_REAL_DIAMETER_CM = 2.3

FOOD_INFO = {
    "kofta":             {"density": 1.05, "calories_per_g": 1.67},
    "macaroni_bechamel": {"density": 0.90, "calories_per_g": 3.00},
    "Chicken":           {"density": 1.00, "calories_per_g": 1.65},
    "mahshi":            {"density": 0.88, "calories_per_g": 1.20},
    "taameya":           {"density": 0.75, "calories_per_g": 2.77},
    "fool":              {"density": 0.92, "calories_per_g": 1.10},
    "molokhia":          {"density": 0.95, "calories_per_g": 0.53},
    "koshary":           {"density": 0.85, "calories_per_g": 1.15},
    "rice":              {"density": 0.80, "calories_per_g": 1.30},
}

# ── Models (loaded once at startup) ──────────────────────────────────────────
_yolo      = None
_midas     = None
_transform = None
_device    = None

def load_models():
    global _yolo, _midas, _transform, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_path = os.path.join(os.path.dirname(__file__), "best.pt")
    _yolo = YOLO(yolo_path)

    _midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _midas.to(_device).eval()
    _transform = torch.hub.load(
        "intel-isl/MiDaS", "transforms", trust_repo=True
    ).small_transform

    print(f" Models loaded on {_device}")


# ── Depth helpers ─────────────────────────────────────────────────────────────
def get_depth_map(image_bgr):
    h, w    = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor  = _transform(img_rgb).to(_device)
    with torch.no_grad():
        depth = _midas(tensor)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def _resize_mask(mask, target_hw):
    """Resize a YOLO mask (H×W float) to target (h, w)."""
    h, w = target_hw
    return cv2.resize(
        mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    )


def get_pixel_size_from_coin(coin_mask, target_hw):
    """Return cm-per-pixel using the coin diameter as reference."""
    mask_uint8  = (_resize_mask(coin_mask, target_hw) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    _, radius = cv2.minEnclosingCircle(largest)
    diameter_pixels = radius * 2
    if diameter_pixels < 5:
        return None
    return COIN_REAL_DIAMETER_CM / diameter_pixels


# ── Calorie estimation ────────────────────────────────────────────────────────
DEPTH_MULTIPLIER = 5.25   # calibrated constant (replaces dynamic depth_scale)

def calc_calories(food_name, food_mask, depth_map, pixel_size_cm):
    if food_name not in FOOD_INFO:
        print(f"    '{food_name}' not in FOOD_INFO")
        return None

    food      = FOOD_INFO[food_name]
    mask_bool = _resize_mask(food_mask, depth_map.shape[:2]).astype(bool)

    area_cm2     = mask_bool.sum() * (pixel_size_cm ** 2)
    avg_depth_cm = float(depth_map[mask_bool].mean()) * DEPTH_MULTIPLIER
    avg_depth_cm = max(1.0, min(avg_depth_cm, 10.0))   # clamp to sane range

    volume_cm3 = area_cm2 * avg_depth_cm
    weight_g   = volume_cm3 * food["density"]
    calories   = weight_g   * food["calories_per_g"]

    return {
        "area_cm2":   round(area_cm2,   2),
        "depth_cm":   round(avg_depth_cm, 2),
        "volume_cm3": round(volume_cm3,  2),
        "weight_g":   round(weight_g,    1),
        "calories":   round(calories,    1),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def predict(image_bgr):
    """
    Args:
        image_bgr: numpy array (H×W×3, BGR) – the raw image.
    Returns:
        dict with keys:
            "items"           – list of per-food dicts
            "total_weight_g"  – float
            "total_calories"  – float
            "annotated_image" – numpy array (BGR) with YOLO masks drawn
    """
    results     = _yolo.predict(image_bgr, conf=0.25, task="segment", verbose=False)
    result      = results[0]
    class_names = _yolo.names

    if result.masks is None:
        return {"items": [], "total_weight_g": 0, "total_calories": 0, "annotated_image": image_bgr}

    depth_map = get_depth_map(image_bgr)
    hw        = image_bgr.shape[:2]

    coin_mask  = None
    food_items = []

    for i in range(len(result.masks)):
        mask       = result.masks.data[i].cpu().numpy()
        class_name = class_names[int(result.boxes.cls[i])]
        confidence = float(result.boxes.conf[i])

        if class_name == COIN_CLASS_NAME:
            coin_mask = mask
        else:
            food_items.append((class_name, mask, confidence))

    if coin_mask is None:
        return {
            "items": [],
            "total_weight_g": 0,
            "total_calories": 0,
            "annotated_image": result.plot(),
            "error": "No coin detected – place a 1-pound coin next to the food.",
        }

    pixel_size_cm = get_pixel_size_from_coin(coin_mask, hw)
    if pixel_size_cm is None:
        return {
            "items": [],
            "total_weight_g": 0,
            "total_calories": 0,
            "annotated_image": result.plot(),
            "error": "Could not measure coin size in image.",
        }

    items          = []
    total_calories = 0.0

    for food_name, mask, confidence in food_items:
        data = calc_calories(food_name, mask, depth_map, pixel_size_cm)
        if data:
            total_calories += data["calories"]
            items.append({
                "name":       food_name,
                "confidence": round(confidence, 2),
                **data,
            })

    return {
        "items":          items,
        "total_weight_g": round(sum(item["weight_g"] for item in items), 1),
        "total_calories": round(total_calories, 1),
        "annotated_image": result.plot(),   # BGR numpy array
    }

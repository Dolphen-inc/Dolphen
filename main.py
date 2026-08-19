import base64
import io
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# ============================================================
# CONFIG
# ============================================================

MAX_IMAGE_DIMENSION = 4096


# ============================================================
# IMAGE IO
# ============================================================

def base64_to_pil(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Prevent unnecessarily huge processing loads
    if max(img.size) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(img.size)
        img = img.resize(
            (
                int(img.width * scale),
                int(img.height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    return img


def pil_to_base64(pil_img):
    buffer = io.BytesIO()

    pil_img = pil_img.convert("RGB")
    pil_img.save(
        buffer,
        format="PNG",
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def pil_to_cv(img):
    return cv2.cvtColor(
        np.array(img),
        cv2.COLOR_RGB2BGR
    )


def cv_to_pil(img):
    return Image.fromarray(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )


def clamp_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# BASIC MASK UTILITIES
# ============================================================

def feather_mask(mask, radius=7):
    """
    Softens a binary/float mask so retouching blends naturally.
    """
    mask = np.clip(mask, 0, 1).astype(np.float32)

    if radius <= 0:
        return mask

    k = max(3, int(radius) * 2 + 1)

    return cv2.GaussianBlur(
        mask,
        (k, k),
        radius
    )


def refine_mask(mask, close_size=3, blur=5):
    """
    Morphological cleanup + feathering.
    """
    mask = (mask > 0.5).astype(np.uint8) * 255

    kernel = np.ones(
        (close_size, close_size),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = mask.astype(np.float32) / 255.0

    return feather_mask(mask, blur)


def blend_with_mask(original, processed, mask):
    """
    Professional-style soft compositing.
    """
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    mask = np.clip(mask, 0, 1)

    if mask.ndim == 2:
        mask = mask[..., None]

    output = (
        original * (1.0 - mask) +
        processed * mask
    )

    return clamp_uint8(output)


# ============================================================
# COLOR SPACE UTILITIES
# ============================================================

def rgb_to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def lab_to_rgb(img):
    return cv2.cvtColor(
        img,
        cv2.COLOR_LAB2RGB
    )


def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


# ============================================================
# SKIN DETECTION
# ============================================================

def skin_mask_rgb(img):
    """
    Multi-condition skin detector.

    Combines:
      - RGB relationship
      - HSV hue/saturation
      - LAB chroma

    This is much less destructive than a simple RGB threshold.
    """

    rgb = img.astype(np.uint8)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    hsv = rgb_to_hsv(rgb)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # RGB skin relationship
    rgb_condition = (
        (r > 60) &
        (g > 30) &
        (b > 15) &
        (r > g) &
        (g > b) &
        ((r - g) > 8)
    )

    # HSV: broad human-skin hue range
    hsv_condition = (
        (h >= 0) &
        (h <= 35) &
        (s >= 20) &
        (s <= 230) &
        (v >= 45)
    )

    # LAB warm/red chroma relationship
    lab = rgb_to_lab(rgb)
    a = lab[:, :, 1].astype(np.float32)
    lab_b = lab[:, :, 2].astype(np.float32)

    lab_condition = (
        (a > 125) &
        (lab_b > 125)
    )

    mask = (
        rgb_condition &
        hsv_condition &
        lab_condition
    )

    # Exclude extremely saturated colors
    mask &= s < 240

    return refine_mask(
        mask.astype(np.float32),
        close_size=5,
        blur=6
    )


# ============================================================
# TEETH MASK
# ============================================================

def teeth_mask(img):
    """
    Detects tooth-like regions based on:
      - low/moderate saturation
      - bright luminance
      - warm/yellow bias
      - avoids highly saturated skin/lip colors
    """

    hsv = rgb_to_hsv(img)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    rgb = img.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    luminance = (
        0.2126 * r +
        0.7152 * g +
        0.0722 * b
    )

    # Teeth tend to be bright and relatively low saturation.
    mask = (
        (luminance > 120) &
        (v > 125) &
        (s < 110) &
        (r > 90) &
        (g > 85) &
        (b > 55) &
        (r >= g * 0.85)
    )

    # Remove strongly colored regions.
    mask &= ~(
        (s > 130) &
        (v < 220)
    )

    return refine_mask(
        mask.astype(np.float32),
        close_size=3,
        blur=4
    )


# ============================================================
# EYE WHITE MASK
# ============================================================

def eye_white_mask(img):
    """
    Finds bright, low-saturation regions suitable for
    sclera/highlight enhancement.
    """

    rgb = img.astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    luminance = (
        0.299 * r +
        0.587 * g +
        0.114 * b
    )

    chroma = (
        np.maximum.reduce([r, g, b]) -
        np.minimum.reduce([r, g, b])
    )

    mask = (
        (luminance > 105) &
        (chroma < 45) &
        (r > 80) &
        (g > 80) &
        (b > 70)
    )

    return refine_mask(
        mask.astype(np.float32),
        close_size=3,
        blur=3
    )


# ============================================================
# LIP MASK
# ============================================================

def lip_mask(img):
    """
    Broad red/pink lip detector using HSV + LAB.
    """

    hsv = rgb_to_hsv(img)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    lab = rgb_to_lab(img)
    a = lab[:, :, 1].astype(np.float32)

    # OpenCV hue: red is around 0 / 180
    red_hue = (
        (h <= 12) |
        (h >= 165)
    )

    pink_red = (
        red_hue &
        (s > 35) &
        (s < 240) &
        (v > 35) &
        (a > 135)
    )

    return refine_mask(
        pink_red.astype(np.float32),
        close_size=5,
        blur=5
    )


# ============================================================
# RED-EYE MASK
# ============================================================

def red_eye_mask(img):
    rgb = img.astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    mask = (
        (r > 90) &
        (r > g * 1.45) &
        (r > b * 1.35) &
        ((r - g) > 25)
    )

    # Red-eye usually appears as a relatively small region.
    mask = mask.astype(np.uint8) * 255

    # Remove large regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if 5 <= area <= 5000:
            clean[labels == i] = 255

    return feather_mask(
        clean.astype(np.float32) / 255.0,
        3
    )


# ============================================================
# PROFESSIONAL SKIN SMOOTHING
# ============================================================

def professional_skin_smoothing(img, strength=0.55):
    """
    Frequency-aware skin smoothing.

    Keeps:
      - eyes
      - lips
      - nostrils
      - major facial edges

    while reducing:
      - pores
      - minor texture
      - small blemishes
    """

    original = img.astype(np.float32)

    skin = skin_mask_rgb(img)

    # Bilateral filtering preserves major edges better than
    # a normal Gaussian blur.
    smooth = cv2.bilateralFilter(
        img,
        d=0,
        sigmaColor=35,
        sigmaSpace=9
    ).astype(np.float32)

    # Additional gentle Gaussian component
    gaussian = cv2.GaussianBlur(
        img,
        (0, 0),
        2.0
    ).astype(np.float32)

    processed = (
        smooth * 0.75 +
        gaussian * 0.25
    )

    mask = skin * strength

    return blend_with_mask(
        original,
        processed,
        mask
    )


# ============================================================
# PROFESSIONAL TEETH WHITENING
# ============================================================

def professional_teeth_whitening(img, strength=0.70):
    """
    Teeth whitening:
      1. Desaturates yellow/chroma
      2. Slightly raises luminance
      3. Preserves tooth texture
      4. Avoids clipping highlights
    """

    original = img.astype(np.float32)

    mask = teeth_mask(img)

    lab = rgb_to_lab(img)

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # Reduce yellow chroma.
    B_new = B - 14.0 * strength

    # Very small neutralization toward center.
    A_new = A + (128.0 - A) * 0.10 * strength

    # Lift luminance carefully.
    L_new = L + 8.0 * strength

    lab_processed = np.stack(
        [
            np.clip(L_new, 0, 255),
            np.clip(A_new, 0, 255),
            np.clip(B_new, 0, 255)
        ],
        axis=2
    ).astype(np.uint8)

    processed = lab_to_rgb(lab_processed).astype(np.float32)

    return blend_with_mask(
        original,
        processed,
        mask
    )


# ============================================================
# PROFESSIONAL EYE BRIGHTENING
# ============================================================

def professional_eye_brightening(img, strength=0.55):
    original = img.astype(np.float32)

    mask = eye_white_mask(img)

    lab = rgb_to_lab(img)

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # Brighten luminance without blowing whites.
    L_new = L + 10.0 * strength

    # Slight neutralization of redness/yellowness.
    A_new = A + (128.0 - A) * 0.12 * strength
    B_new = B + (128.0 - B) * 0.08 * strength

    processed_lab = np.stack(
        [
            np.clip(L_new, 0, 255),
            np.clip(A_new, 0, 255),
            np.clip(B_new, 0, 255)
        ],
        axis=2
    ).astype(np.uint8)

    processed = lab_to_rgb(processed_lab).astype(np.float32)

    return blend_with_mask(
        original,
        processed,
        mask * strength
    )


# ============================================================
# LIP TINT
# ============================================================

def professional_lip_tint(img, strength=0.45):
    """
    Enhances existing lip color rather than painting arbitrary
    red pixels across the image.
    """

    original = img.astype(np.float32)

    mask = lip_mask(img)

    hsv = rgb_to_hsv(img)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Increase saturation moderately.
    s_new = s * (
        1.0 + 0.28 * strength
    )

    # Slight hue movement toward a natural pink/red.
    target_hue = 175.0

    hue_distance = (
        (target_hue - h + 90) % 180
    ) - 90

    h_new = h + hue_distance * 0.10 * strength

    hsv_processed = np.stack(
        [
            np.clip(h_new, 0, 179),
            np.clip(s_new, 0, 255),
            np.clip(v, 0, 255)
        ],
        axis=2
    ).astype(np.uint8)

    processed = cv2.cvtColor(
        hsv_processed,
        cv2.COLOR_HSV2RGB
    ).astype(np.float32)

    return blend_with_mask(
        original,
        processed,
        mask
    )


# ============================================================
# RED EYE CORRECTION
# ============================================================

def professional_red_eye_correction(img):
    original = img.astype(np.float32)

    mask = red_eye_mask(img)

    if mask.max() <= 0:
        return img

    rgb = img.astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Red-eye pupils should generally be dark/neutral.
    neutral = (
        0.45 * g +
        0.45 * b +
        0.10 * r
    )

    processed = rgb.copy()

    processed[:, :, 0] = (
        r * 0.20 +
        neutral * 0.80
    )

    processed[:, :, 1] = (
        g * 0.70 +
        neutral * 0.30
    )

    processed[:, :, 2] = (
        b * 0.70 +
        neutral * 0.30
    )

    return blend_with_mask(
        original,
        processed,
        mask
    )


# ============================================================
# SHARPENING BRUSH
# ============================================================

def professional_sharpen(img, amount=1.0):
    """
    High-frequency sharpening with edge protection.
    """

    original = img.astype(np.float32)

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.25
    ).astype(np.float32)

    high_frequency = original - blur

    # Edge strength
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Laplacian(
        gray,
        cv2.CV_32F
    )

    edge_strength = np.abs(edges)

    edge_strength = cv2.normalize(
        edge_strength,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    # Avoid extreme sharpening in flat/noisy areas.
    mask = np.clip(
        edge_strength * 1.5,
        0,
        1
    )

    mask = feather_mask(mask, 1.5)

    processed = (
        original +
        high_frequency * (0.65 * amount)
    )

    return blend_with_mask(
        original,
        processed,
        mask
    )


# ============================================================
# NOISE REDUCTION
# ============================================================

def professional_noise_reduction(img, strength=0.55):
    """
    Edge-aware denoising using bilateral filtering.
    """

    original = img.astype(np.float32)

    filtered = cv2.bilateralFilter(
        img,
        d=7,
        sigmaColor=35,
        sigmaSpace=35
    ).astype(np.float32)

    processed = (
        original * (1.0 - strength) +
        filtered * strength
    )

    return clamp_uint8(processed)


# ============================================================
# FREQUENCY SEPARATION
# ============================================================

def frequency_separation(img, radius=8, texture_strength=0.35):
    """
    Professional frequency-separation style processing.

    Low frequency = color/tone
    High frequency = texture/detail

    Reduces uneven skin tone while retaining texture.
    """

    original = img.astype(np.float32)

    low = cv2.GaussianBlur(
        original,
        (0, 0),
        radius
    )

    high = original - low

    # Suppress only some high frequency.
    high_processed = high * (
        1.0 - texture_strength
    )

    processed = low + high_processed

    return clamp_uint8(processed)


# ============================================================
# BLEMISH REMOVER
# ============================================================

def professional_blemish_remover(img):
    """
    Automatic small blemish reduction.

    Detects isolated high-frequency spots and repairs them
    using surrounding pixels.

    This intentionally avoids large structures.
    """

    original = img.copy()

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        3
    )

    difference = cv2.absdiff(
        gray,
        smooth
    )

    # Strong local deviations.
    mask = difference > 18

    # Only small connected components.
    mask_uint = mask.astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint,
        8
    )

    clean = np.zeros_like(mask_uint)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if 4 <= area <= 250:
            clean[labels == i] = 255

    clean = cv2.dilate(
        clean,
        np.ones((3, 3), np.uint8)
    )

    clean = feather_mask(
        clean.astype(np.float32) / 255.0,
        3
    )

    # Edge-preserving reconstruction.
    repaired = cv2.bilateralFilter(
        img,
        7,
        45,
        45
    )

    return blend_with_mask(
        original.astype(np.float32),
        repaired.astype(np.float32),
        clean * 0.65
    )


# ============================================================
# DODGE & BURN
# ============================================================

def dodge_and_burn(img, mode="dodge", strength=0.18):
    """
    Global tonal dodge/burn operation.

    mode:
      dodge = brighten
      burn  = darken
    """

    original = img.astype(np.float32) / 255.0

    if mode == "dodge":
        processed = original + (
            (1.0 - original) * strength
        )
    else:
        processed = original * (
            1.0 - strength
        )

    return clamp_uint8(
        processed * 255.0
    )


# ============================================================
# MATTE DEFRINGE
# ============================================================

def matte_defringe(img, strength=0.65):
    """
    Reduces chromatic fringing around high-contrast edges.
    """

    original = img.astype(np.float32)

    lab = rgb_to_lab(img)

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # Smooth chroma only.
    A_blur = cv2.GaussianBlur(
        A,
        (0, 0),
        1.5
    )

    B_blur = cv2.GaussianBlur(
        B,
        (0, 0),
        1.5
    )

    A_new = A * (1.0 - strength) + A_blur * strength
    B_new = B * (1.0 - strength) + B_blur * strength

    processed_lab = np.stack(
        [
            L,
            A_new,
            B_new
        ],
        axis=2
    ).astype(np.uint8)

    processed = lab_to_rgb(
        processed_lab
    ).astype(np.float32)

    # Apply mainly around edges.
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        60,
        140
    )

    edge_mask = feather_mask(
        edges.astype(np.float32) / 255.0,
        2
    )

    return blend_with_mask(
        original,
        processed,
        edge_mask * strength
    )


# ============================================================
# SKIN TONE BALANCER
# ============================================================

def skin_tone_balancer(img, strength=0.40):
    """
    Normalizes uneven skin chroma while preserving luminance.
    """

    original = img.astype(np.float32)

    mask = skin_mask_rgb(img)

    lab = rgb_to_lab(img)

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # Estimate local chroma averages.
    a_blur = cv2.GaussianBlur(
        A,
        (0, 0),
        12
    )

    b_blur = cv2.GaussianBlur(
        B,
        (0, 0),
        12
    )

    A_new = (
        A * (1.0 - strength) +
        a_blur * strength
    )

    B_new = (
        B * (1.0 - strength) +
        b_blur * strength
    )

    processed_lab = np.stack(
        [
            L,
            A_new,
            B_new
        ],
        axis=2
    ).astype(np.uint8)

    processed = lab_to_rgb(
        processed_lab
    ).astype(np.float32)

    return blend_with_mask(
        original,
        processed,
        mask
    )


# ============================================================
# CLONE STAMP
# ============================================================

def clone_stamp(img, source_x, source_y, target_x, target_y, radius):
    """
    Clone a circular source region onto a target region.

    Coordinates are pixel coordinates.
    """

    result = img.copy()

    h, w = img.shape[:2]

    radius = int(max(1, radius))

    y1 = max(0, target_y - radius)
    y2 = min(h, target_y + radius)

    x1 = max(0, target_x - radius)
    x2 = min(w, target_x + radius)

    source_x1 = source_x - (target_x - x1)
    source_x2 = source_x + (x2 - target_x)

    source_y1 = source_y - (target_y - y1)
    source_y2 = source_y + (y2 - target_y)

    if (
        source_x1 < 0 or
        source_y1 < 0 or
        source_x2 > w or
        source_y2 > h
    ):
        return result

    patch = img[
        source_y1:source_y2,
        source_x1:source_x2
    ].copy()

    target = result[
        y1:y2,
        x1:x2
    ]

    ph, pw = patch.shape[:2]

    yy, xx = np.mgrid[
        0:ph,
        0:pw
    ]

    cx = pw / 2
    cy = ph / 2

    distance = np.sqrt(
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    )

    alpha = np.clip(
        1.0 - distance / radius,
        0,
        1
    )

    alpha = cv2.GaussianBlur(
        alpha.astype(np.float32),
        (0, 0),
        max(1, radius * 0.12)
    )

    alpha = alpha[..., None]

    result[
        y1:y2,
        x1:x2
    ] = (
        target.astype(np.float32) * (1 - alpha) +
        patch.astype(np.float32) * alpha
    ).astype(np.uint8)

    return result


# ============================================================
# OBJECT ERASER
# ============================================================

def object_eraser(img, mask):
    """
    Content-aware object removal.

    The frontend should send a grayscale mask in the same
    dimensions as the source image.
    """

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_RGB2GRAY
        )

    mask = (mask > 127).astype(np.uint8) * 255

    # Expand slightly to ensure edges are removed.
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    result = cv2.inpaint(
        img,
        mask,
        5,
        cv2.INPAINT_TELEA
    )

    return result


# ============================================================
# RESHAPE TOOL
# ============================================================

def reshape_image(img, displacement_map):
    """
    Mesh/displacement-based reshaping.

    displacement_map must be supplied by the frontend as:

        {
            "dx": [...],
            "dy": [...]
        }

    Both arrays must match image dimensions.

    This gives the frontend precise control over:
      - face slimming
      - jaw shaping
      - waist shaping
      - eye enlargement
      - nose shaping
    """

    h, w = img.shape[:2]

    dx = np.asarray(
        displacement_map["dx"],
        dtype=np.float32
    )

    dy = np.asarray(
        displacement_map["dy"],
        dtype=np.float32
    )

    if dx.shape != (h, w) or dy.shape != (h, w):
        raise ValueError(
            "Displacement maps must match image dimensions"
        )

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = x - dx
    map_y = y - dy

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )


# ============================================================
# DIGITAL MAKEUP
# ============================================================

def digital_makeup(img, makeup):
    """
    Makeup operations based on masks supplied by frontend.

    Supported:
      lipstick
      blush
      eyeshadow
      eyeliner
    """

    result = img.astype(np.float32)

    for layer in makeup:

        layer_type = layer.get("type")
        color = np.array(
            layer.get("color", [255, 0, 0]),
            dtype=np.float32
        )

        opacity = float(
            layer.get("opacity", 0.35)
        )

        mask = np.asarray(
            layer.get("mask"),
            dtype=np.float32
        )

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = np.clip(
            mask / 255.0,
            0,
            1
        )

        mask = feather_mask(
            mask,
            layer.get("feather", 5)
        )

        # Makeup should blend with underlying luminosity.
        if layer_type in (
            "lipstick",
            "blush",
            "eyeshadow"
        ):
            color_layer = np.zeros_like(result)
            color_layer[:, :] = color

            blend = (
                result * 0.55 +
                color_layer * 0.45
            )

            alpha = mask * opacity

            result = (
                result * (1 - alpha[..., None]) +
                blend * alpha[..., None]
            )

        elif layer_type == "eyeliner":
            color_layer = np.zeros_like(result)
            color_layer[:, :] = color

            alpha = mask * opacity

            result = (
                result * (1 - alpha[..., None]) +
                color_layer * alpha[..., None]
            )

    return clamp_uint8(result)


# ============================================================
# REQUEST PARSING
# ============================================================

def get_strength(data, default=0.5):
    try:
        return float(
            np.clip(
                data.get("strength", default),
                0,
                1
            )
        )
    except Exception:
        return default


# ============================================================
# MAIN RETOUCH PROCESSOR
# ============================================================

def process_retouch_task(task_name):
    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({
            "error": "Image data missing"
        }), 400

    try:
        pil_img = base64_to_pil(
            data["image"]
        )

        img = np.array(
            pil_img,
            dtype=np.uint8
        )

        strength = get_strength(
            data,
            0.55
        )

        # ====================================================
        # SKIN SMOOTHER
        # ====================================================

        if task_name == "skinsmoother":

            result = professional_skin_smoothing(
                img,
                strength
            )

        # ====================================================
        # TEETH WHITENER
        # ====================================================

        elif task_name == "teethwhitener":

            result = professional_teeth_whitening(
                img,
                strength
            )

        # ====================================================
        # EYE BRIGHTENER
        # ====================================================

        elif task_name == "eyebrightener":

            result = professional_eye_brightening(
                img,
                strength
            )

        # ====================================================
        # LIP TINT
        # ====================================================

        elif task_name == "liptintoverlay":

            result = professional_lip_tint(
                img,
                strength
            )

        # ====================================================
        # RED EYE
        # ====================================================

        elif task_name == "redeyecorrector":

            result = professional_red_eye_correction(
                img
            )

        # ====================================================
        # SHARPENING
        # ====================================================

        elif task_name == "sharpeningbrush":

            result = professional_sharpen(
                img,
                amount=0.5 + strength
            )

        # ====================================================
        # NOISE REDUCTION
        # ====================================================

        elif task_name == "noisereducer":

            result = professional_noise_reduction(
                img,
                strength
            )

        # ====================================================
        # FREQUENCY SEPARATION
        # ====================================================

        elif task_name == "frequencyseparation":

            result = frequency_separation(
                img,
                radius=float(
                    data.get("radius", 8)
                ),
                texture_strength=strength
            )

        # ====================================================
        # BLEMISH REMOVER
        # ====================================================

        elif task_name == "blemishremover":

            result = professional_blemish_remover(
                img
            )

        # ====================================================
        # SKIN TONE BALANCER
        # ====================================================

        elif task_name == "skintonebalancer":

            result = skin_tone_balancer(
                img,
                strength
            )

        # ====================================================
        # MATTE DEFRINGE
        # ====================================================

        elif task_name == "mattedefringe":

            result = matte_defringe(
                img,
                strength
            )

        # ====================================================
        # DODGE & BURN
        # ====================================================

        elif task_name == "dodgeandburn":

            mode = data.get(
                "mode",
                "dodge"
            )

            result = dodge_and_burn(
                img,
                mode=mode,
                strength=0.05 + strength * 0.25
            )

        # ====================================================
        # CLONE STAMP
        # ====================================================

        elif task_name == "clonestamp":

            required = [
                "sourceX",
                "sourceY",
                "targetX",
                "targetY",
                "radius"
            ]

            if not all(
                key in data
                for key in required
            ):
                return jsonify({
                    "error": (
                        "Clone stamp requires "
                        "sourceX, sourceY, targetX, "
                        "targetY and radius"
                    )
                }), 400

            result = clone_stamp(
                img,
                int(data["sourceX"]),
                int(data["sourceY"]),
                int(data["targetX"]),
                int(data["targetY"]),
                int(data["radius"])
            )

        # ====================================================
        # OBJECT ERASER
        # ====================================================

        elif task_name == "objectEraser":

            if "mask" not in data:
                return jsonify({
                    "error": (
                        "Object eraser requires "
                        "a mask"
                    )
                }), 400

            mask_img = base64_to_pil(
                data["mask"]
            )

            mask = np.array(
                mask_img
            )

            result = object_eraser(
                img,
                mask
            )

        # ====================================================
        # RESHAPE
        # ====================================================

        elif task_name == "reshapetool":

            if "displacementMap" not in data:
                return jsonify({
                    "error": (
                        "Reshape tool requires "
                        "a displacementMap"
                    )
                }), 400

            result = reshape_image(
                img,
                data["displacementMap"]
            )

        # ====================================================
        # DIGITAL MAKEUP
        # ====================================================

        elif task_name == "digitalmakeupcanvas":

            makeup = data.get(
                "makeup",
                []
            )

            result = digital_makeup(
                img,
                makeup
            )

        # ====================================================
        # FALLBACK
        # ====================================================

        else:

            result = img

        output = pil_to_base64(
            Image.fromarray(result)
        )

        return jsonify({
            "status": "success",
            "task": task_name,
            "processedImageUrl": output
        }), 200

    except Exception as e:

        print(
            f"[ERROR] Retouch failed: {str(e)}"
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# ENDPOINTS
# ============================================================

@app.route(
    "/api/retouch/teethwhitener",
    methods=["POST"]
)
def teethwhitener():
    return process_retouch_task(
        "teethwhitener"
    )


@app.route(
    "/api/retouch/eyebrightener",
    methods=["POST"]
)
def eyebrightener():
    return process_retouch_task(
        "eyebrightener"
    )


@app.route(
    "/api/retouch/sharpeningbrush",
    methods=["POST"]
)
def sharpeningbrush():
    return process_retouch_task(
        "sharpeningbrush"
    )


@app.route(
    "/api/retouch/reshapetool",
    methods=["POST"]
)
def reshapetool():
    return process_retouch_task(
        "reshapetool"
    )


@app.route(
    "/api/retouch/frequencyseparation",
    methods=["POST"]
)
def frequencyseparation():
    return process_retouch_task(
        "frequencyseparation"
    )


@app.route(
    "/api/retouch/clonestamp",
    methods=["POST"]
)
def clonestamp():
    return process_retouch_task(
        "clonestamp"
    )


@app.route(
    "/api/retouch/dodgeandburn",
    methods=["POST"]
)
def dodgeandburn():
    return process_retouch_task(
        "dodgeandburn"
    )


@app.route(
    "/api/retouch/mattedefringe",
    methods=["POST"]
)
def mattedefringe():
    return process_retouch_task(
        "mattedefringe"
    )


@app.route(
    "/api/retouch/skintonebalancer",
    methods=["POST"]
)
def skintonebalancer():
    return process_retouch_task(
        "skintonebalancer"
    )


@app.route(
    "/api/retouch/noisereducer",
    methods=["POST"]
)
def noisereducer():
    return process_retouch_task(
        "noisereducer"
    )


@app.route(
    "/api/retouch/liptintoverlay",
    methods=["POST"]
)
def liptintoverlay():
    return process_retouch_task(
        "liptintoverlay"
    )


@app.route(
    "/api/retouch/digitalmakeupcanvas",
    methods=["POST"]
)
def digitalmakeupcanvas():
    return process_retouch_task(
        "digitalmakeupcanvas"
    )


@app.route(
    "/api/retouch/objectEraser",
    methods=["POST"]
)
def object_eraser_endpoint():
    return process_retouch_task(
        "objectEraser"
    )


@app.route(
    "/api/retouch/redeyecorrector",
    methods=["POST"]
)
def redeyecorrector():
    return process_retouch_task(
        "redeyecorrector"
    )


@app.route(
    "/api/retouch/blemishremover",
    methods=["POST"]
)
def blemishremover():
    return process_retouch_task(
        "blemishremover"
    )


@app.route(
    "/api/retouch/skinsmoother",
    methods=["POST"]
)
def skinsmoother():
    return process_retouch_task(
        "skinsmoother"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "professional-retouch-backend"
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

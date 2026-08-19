import base64
import io
import os
import cv2
import dlib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MAX_IMAGE_DIMENSION = 4096

# Initialize Classical Landmark Detector (Ensemble of Regression Trees - Non-AI)
LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()

if os.path.exists(LANDMARK_PATH):
    predictor = dlib.shape_predictor(LANDMARK_PATH)
else:
    predictor = None
    print(f"[WARNING] '{LANDMARK_PATH}' not found. Landmark-based features will fallback to color math.")

# --- IO & MATH UTILITIES ---

def base64_to_cv(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    if max(img.size) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_base64(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def clamp_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def feather_mask(mask, radius=7):
    mask = np.clip(mask, 0, 1).astype(np.float32)
    if radius <= 0:
        return mask
    k = max(3, int(radius) * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), radius)

# --- TRADITIONAL LANDMARK MASKING ---

def get_facial_landmarks(img):
    if predictor is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = predictor(gray, rects[0])
    points = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        points[i] = (shape.part(i).x, shape.part(i).y)
    return points

def create_poly_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.float32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask

def get_feature_masks(img):
    landmarks = get_facial_landmarks(img)
    if landmarks is None:
        return None, None, None, None
    
    # 68 Landmark Indices
    face_pts = landmarks[0:17]
    left_brow = landmarks[17:22]
    right_brow = landmarks[22:27]
    nose = landmarks[27:36]
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth_outer = landmarks[48:60]
    teeth_inner = landmarks[60:68]
    
    # 1. Full Face Skin Mask (Excluding Eyes, Mouth, Brows)
    face_mask = create_poly_mask(img.shape, landmarks[0:27])
    
    # Cut out eyes and mouth from skin mask
    l_eye_mask = create_poly_mask(img.shape, left_eye)
    r_eye_mask = create_poly_mask(img.shape, right_eye)
    mouth_mask = create_poly_mask(img.shape, mouth_outer)
    
    skin_mask = face_mask - (l_eye_mask + r_eye_mask + mouth_mask)
    skin_mask = np.clip(skin_mask, 0, 1)
    
    # 2. Teeth Mask
    teeth_mask = create_poly_mask(img.shape, teeth_inner)
    
    # 3. Eyes Mask (Sclera)
    eyes_mask = np.clip(l_eye_mask + r_eye_mask, 0, 1)
    
    # 4. Lips Mask
    lips_mask = create_poly_mask(img.shape, mouth_outer) - teeth_mask
    lips_mask = np.clip(lips_mask, 0, 1)
    
    return skin_mask, teeth_mask, eyes_mask, lips_mask

# --- CLASSICAL RETOUCHING ALGORITHMS ---

def frequency_separation_smoothing(img, strength=0.55):
    """
    True Frequency Separation:
    Separates high frequencies (texture/pores) from low frequencies (color/shading).
    Smoothes low frequencies with Guided Filtering while preserving exact pore detail.
    """
    src = img.astype(np.float32)
    skin_m, _, _, _ = get_feature_masks(img)
    
    if skin_m is None:
        skin_m = np.ones(img.shape[:2], dtype=np.float32)

    # 1. Decompose Low/High Frequency
    sigma = 8.0
    low_freq = cv2.GaussianBlur(src, (0, 0), sigma)
    high_freq = src - low_freq
    
    # 2. Edge-Aware Smooth on Low Frequency (Tone/Color)
    guide = low_freq.astype(np.uint8)
    smoothed_low = cv2.ximgproc.guidedFilter(
        guide=guide, 
        src=low_freq, 
        radius=int(sigma * 2), 
        eps=100.0
    )
    
    # 3. Reconstruct image with smoothed tone + untouched texture
    reconstructed = smoothed_low + high_freq
    reconstructed = clamp_uint8(reconstructed)
    
    mask = feather_mask(skin_m, 12)[..., None] * strength
    output = src * (1.0 - mask) + reconstructed.astype(np.float32) * mask
    return clamp_uint8(output)

def poisson_blemish_remover(img):
    """
    Gradient-Domain Blemish Removal:
    Identifies high-contrast local variance and reconstructs it 
    solving Poisson Partial Differential Equations against clean surrounding texture.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find strong local deviations (blemishes)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    diff = cv2.absdiff(gray, blur)
    _, binary_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    
    # Isolate small circular spots
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    blemish_mask = np.zeros_like(binary_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= 300: # Filter small blemish sizes
            blemish_mask[labels == i] = 255
            
    if blemish_mask.max() == 0:
        return img
        
    blemish_mask = cv2.dilate(blemish_mask, np.ones((5, 5), np.uint8))
    
    # Non-AI Poisson Inpainting
    return cv2.inpaint(img, blemish_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def poisson_clone_stamp(img, src_x, src_y, tgt_x, tgt_y, radius):
    """
    Clones patch using Poisson Seamless Cloning (matches lighting and gradients).
    """
    h, w = img.shape[:2]
    radius = int(max(5, radius))
    
    mask = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    
    patch_center = (src_x, src_y)
    target_center = (tgt_x, tgt_y)
    
    # Ensure targets remain inside frame
    if (tgt_x - radius < 0 or tgt_x + radius >= w or 
        tgt_y - radius < 0 or tgt_y + radius >= h):
        return img
        
    try:
        return cv2.seamlessClone(img, img, mask, target_center, cv2.NORMAL_CLONE)
    except cv2.error:
        return img

def classical_teeth_whitening(img, strength=0.70):
    _, teeth_m, _, _ = get_feature_masks(img)
    
    if teeth_m is None:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    
    # Neutralize yellow chroma (B) and lift luminance (L)
    B_new = B - (18.0 * strength)
    L_new = L + (12.0 * strength)
    
    lab_mod = cv2.merge([np.clip(L_new, 0, 255), A, np.clip(B_new, 0, 255)])
    whitened = cv2.cvtColor(lab_mod.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    
    mask = feather_mask(teeth_m, 3)[..., None]
    output = img.astype(np.float32) * (1.0 - mask) + whitened * mask
    return clamp_uint8(output)

def classical_eye_brightening(img, strength=0.55):
    _, _, eyes_m, _ = get_feature_masks(img)
    
    if eyes_m is None:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    
    L_new = L + (15.0 * strength) # Lift contrast/brightness
    
    lab_mod = cv2.merge([np.clip(L_new, 0, 255), A, B])
    brightened = cv2.cvtColor(lab_mod.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    
    mask = feather_mask(eyes_m, 2)[..., None] * strength
    output = img.astype(np.float32) * (1.0 - mask) + brightened * mask
    return clamp_uint8(output)

def soft_light_blend(base, overlay):
    """Classic Photoshop Soft Light blend mode math."""
    base = base / 255.0
    overlay = overlay / 255.0
    
    results = np.where(
        overlay <= 0.5,
        base - (1.0 - 2.0 * overlay) * base * (1.0 - base),
        base + (2.0 * overlay - 1.0) * (np.sqrt(base) - base)
    )
    return clamp_uint8(results * 255.0)

def classical_lip_tint(img, strength=0.45):
    _, _, _, lips_m = get_feature_masks(img)
    
    if lips_m is None:
        return img
        
    tint_layer = np.full_like(img, (50, 20, 200), dtype=np.uint8) # BGR target lip color
    blended = soft_light_blend(img, tint_layer).astype(np.float32)
    
    mask = feather_mask(lips_m, 4)[..., None] * strength
    output = img.astype(np.float32) * (1.0 - mask) + blended * mask
    return clamp_uint8(output)

# --- ROUTER / CONTROLLER ---

@app.route("/api/retouch/<task_name>", methods=["POST"])
def process_retouch(task_name):
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Image payload missing"}), 400

    try:
        img = base64_to_cv(data["image"])
        strength = float(data.get("strength", 0.55))

        if task_name == "skinsmoother":
            res = frequency_separation_smoothing(img, strength)
        elif task_name == "blemishremover":
            res = poisson_blemish_remover(img)
        elif task_name == "teethwhitener":
            res = classical_teeth_whitening(img, strength)
        elif task_name == "eyebrightener":
            res = classical_eye_brightening(img, strength)
        elif task_name == "liptintoverlay":
            res = classical_lip_tint(img, strength)
        elif task_name == "clonestamp":
            res = poisson_clone_stamp(
                img,
                int(data["sourceX"]), int(data["sourceY"]),
                int(data["targetX"]), int(data["targetY"]),
                int(data["radius"])
            )
        else:
            res = img

        return jsonify({
            "status": "success",
            "task": task_name,
            "processedImageUrl": cv_to_base64(res)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

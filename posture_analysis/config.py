"""
config.py
Centralized configuration for the Interview Posture & Confidence Analyzer.
"""

# =============================================================================
# 1. TEMPORAL & SMOOTHING FACTORS
# =============================================================================
# EMA Alpha determines how quickly values update. 
# 1.0 = instant (no smoothing), 0.0 = never updates.

# Posture — slower alpha = smoother, more accurate tracking
POSTURE_COORD_EMA_ALPHA = 0.3         # was 0.4: slower for less jitter on coords
POSTURE_SCORE_EMA_ALPHA = 0.18        # was 0.25: slower to prevent score swings

# Face & Gaze
FACE_COORD_EMA_ALPHA = 0.35           # slightly faster for better landmark tracking
FACE_SCORE_EMA_ALPHA = 0.25           # was 0.15: faster response so head/eye scores drop when they should
FACE_POSE_EMA_ALPHA = 0.35            # was 0.25: faster tracking of actual head movements
EXPRESSION_SCORE_EMA_ALPHA = 0.04     # slow for expression (stable)

# Hands
HAND_COORD_EMA_ALPHA = 0.35           # was 0.5: less noise on hand coords
HAND_SCORE_EMA_ALPHA = 0.12           # was 0.15: smoother gesture score

# Overall System Scoring
OVERALL_SCORE_EMA_ALPHA = 0.10        # was 0.15: more stable overall readout
RELIABILITY_EMA_ALPHA = 0.20          # was 0.25: less flickery reliability

# =============================================================================
# 2. CALIBRATION CONSTANTS
# =============================================================================
CALIBRATION_FRAMES = 120              # was 90: 4 seconds for more robust baselines
CALIBRATION_WARMUP_SKIP = 15          # NEW: skip first N frames (noisy initial detections)
STD_MULTIPLIER_SLIGHT = 2.0           # was 3.0: tighter tolerance to detect real head turns
STD_MULTIPLIER_MODERATE = 3.5         # was 5.0: narrower band for moderate deviations
STD_MULTIPLIER_EXCESSIVE = 5.0        # was 7.0
CALIBRATION_OUTLIER_TRIM = 0.10       # NEW: trim 10% extreme values from calibration (robust median)

# =============================================================================
# 3. POSTURE THRESHOLDS
# =============================================================================
# Fallback thresholds if variance during calibration is near zero.
SHOULDER_TILT_SLIGHT = 8.0            # degrees of shoulder asymmetry
SHOULDER_TILT_SIGNIFICANT = 18.0      # degrees

# Lateral (Side-to-Side) Lean (degrees in coronal image plane)
LATERAL_LEAN_SLIGHT = 10.0            # degrees of torso lean from vertical
LATERAL_LEAN_MODERATE = 18.0          # degrees
LATERAL_LEAN_EXCESSIVE = 28.0         # degrees
LEAN_SLIGHT = LATERAL_LEAN_SLIGHT     # backward compatibility alias
LEAN_MODERATE = LATERAL_LEAN_MODERATE
LEAN_EXCESSIVE = LATERAL_LEAN_EXCESSIVE

# Forward Lean — Proximity ratio (Current shoulder width / Calibrated baseline width)
# Camera perspective expansion: leaning forward brings shoulders closer to webcam
FORWARD_PROXIMITY_SLIGHT = 1.08       # 8% width expansion
FORWARD_PROXIMITY_MODERATE = 1.16     # 16% width expansion
FORWARD_PROXIMITY_EXCESSIVE = 1.28    # 28% width expansion

# Backward Lean — Proximity ratio (Leaning back away from webcam)
BACKWARD_PROXIMITY_SLIGHT = 0.92      # 8% width shrinkage
BACKWARD_PROXIMITY_MODERATE = 0.84    # 16% width shrinkage

# Slouching / Hunching — Neck vertical compression ratio
# (baseline_neck_ratio - current_neck_ratio) / baseline_neck_ratio
# Neck ratio = (mid_shoulder_y - nose_y) / shoulder_width
SLOUCH_NECK_SLIGHT = 0.10             # 10% vertical neck foreshortening
SLOUCH_NECK_MODERATE = 0.18           # 18% vertical neck foreshortening
SLOUCH_NECK_EXCESSIVE = 0.28          # 28% vertical neck foreshortening

# Slouching — Torso vertical foreshortening (when hips are visible)
SLOUCH_TORSO_SLIGHT = 0.10            # 10% torso height foreshortening
SLOUCH_TORSO_MODERATE = 0.18          # 18% torso height foreshortening

# Shoulder Tension & Shrugging Detection (Elevated Trapezius / Neck Shortening)
SHOULDER_TENSION_RATIO_DROP = 0.18    # 18% reduction in ear-to-shoulder vertical distance
SHOULDER_TENSION_FRAMES = 15          # ~0.5s of sustained tension before flagging

# Head Slump — Vertical downward drop of nose normalized by baseline shoulder width
HEAD_SLUMP_SLIGHT = 0.08              # downward sink in chair
HEAD_SLUMP_MODERATE = 0.15

# Forward Head / Shoulder 3D Depth shifts (relative to baseline)
FORWARD_SHOULDER_Z_SLIGHT = -0.04     # MediaPipe pose z units closer to camera
FORWARD_SHOULDER_Z_MODERATE = -0.08
FORWARD_HEAD_Z_THRESHOLD = -0.06      # ear vs shoulder z shift

HEAD_OFFSET_SLIGHT = 0.40             # horizontal offset normalized by shoulder width
HEAD_OFFSET_SIGNIFICANT = 0.70

POSTURE_STABILITY_WINDOW = 45         # frames for stability measurement
POSTURE_STABILITY_VARIANCE_SCALE = 40000.0  # maps variance to 0-100 penalty
POSTURE_LEAN_HISTORY_FRAMES = 90      # 3 seconds for persistence detection

# Multi-frame consensus for posture status (reduces flicker)
POSTURE_CONSENSUS_WINDOW = 5          # require N consistent frames before status change

# =============================================================================
# 4. FACE & GAZE THRESHOLDS
# =============================================================================
# Fallback thresholds
YAW_SLIGHT = 10.0                     # was 18.0: tighter — 10° yaw is a visible head turn
YAW_SIGNIFICANT = 25.0                # was 40.0: 25° is clearly looking away
PITCH_SLIGHT = 10.0                   # was 18.0: 10° pitch is noticeable
PITCH_SIGNIFICANT = 22.0              # was 35.0: 22° is looking well above/below camera
ROLL_SLIGHT = 10.0                    # was 15.0: tighter head tilt detection
ROLL_SIGNIFICANT = 22.0               # was 30.0

IRIS_CENTER_TOLERANCE = 0.18          # was 0.30: much tighter — detects iris glances away from center

EYE_CONTACT_WINDOW = 45               # was 90: 1.5s window — faster response to gaze changes
GAZE_HYSTERESIS_FRAMES = 4            # was 8: fewer frames before declaring gaze lost
SUSTAINED_AWAY_FRAMES = 45            # was 75: 1.5s to be "Sustained Away"
BLINK_EAR_THRESHOLD = 0.18            # blink detection threshold
BLINK_HOLD_FRAMES = 2                 # was 4: shorter blink forgiveness
BLINK_WINDOW_SEC = 60.0               # rolling window (seconds) for blink rate calculation
BLINK_MIN_DURATION = 0.06             # seconds: minimum human blink duration
BLINK_MAX_DURATION = 0.45             # seconds: maximum human blink duration (longer is closure)
BLINK_LOW = 10.0                      # blinks/min: below this indicates low blink / staring
BLINK_ELEVATED = 35.0                 # blinks/min: above this indicates anxiety / elevated stress

# Multi-frame consensus for gaze status
GAZE_CONSENSUS_WINDOW = 4             # was 6: faster consensus for gaze state changes

# Head pose outlier rejection
HEAD_POSE_MAX_JUMP = 20.0             # was 25.0: tighter rejection of solvePnP glitches

# =============================================================================
# 5. HAND & GESTURE THRESHOLDS
# =============================================================================
HAND_VARIANCE_WINDOW = 120            # was 90: 4 sec for more reliable gesture analysis
FACE_TOUCH_PERSISTENCE = 20           # was 15: require more frames to confirm face touch
FACE_TOUCH_DISTANCE = 0.12            # NEW: was hardcoded 0.15, tighter = fewer false positives
GESTURE_AMPLITUDE_THRESHOLD = 0.18    # was 0.15: slightly higher bar for "Natural Gesture"
FIDGET_FREQ_THRESHOLD = 3.0           # was 2.5: require more direction changes for fidget
FIDGET_AMPLITUDE_MIN = 0.03           # was 0.02: ignore very tiny jitters
FIDGET_AMPLITUDE_MAX = 0.12           # was 0.15: narrower fidget band
FIDGET_CONSENSUS_FRAMES = 15          # NEW: require 0.5s of consistent fidgeting

# Obscene gesture detection
OBSCENE_GESTURE_PERSISTENCE = 10      # NEW: require N frames to confirm (avoids false positives)

# =============================================================================
# 6. SCORING WEIGHTS & PROFILES
# =============================================================================
# DOCUMENTED DISCREPANCY RESOLUTION:
# - Profile "code_default" (from config.py):
#     Posture (25%), Gestures (25%), Eye Contact (20%), Head Position (15%), Expression (10%), Stability (5%)
#     Emphasizes hand composure and fidgeting detection for interview realism.
# - Profile "readme_balanced" (from original visual README.md):
#     Posture (25%), Eye Contact (20%), Head Position (20%), Expression (15%), Stability (10%), Gestures (10%)
#     Distributes weights more evenly across facial orientation and expression.
#
# Default is 'code_default' to preserve the latest calibrated behavior.

SCORING_PROFILES = {
    "code_default": {
        "posture": 0.25,
        "gestures": 0.25,
        "eye_contact": 0.20,
        "head_position": 0.15,
        "expression": 0.10,
        "stability": 0.05,
    },
    "readme_balanced": {
        "posture": 0.25,
        "gestures": 0.10,
        "eye_contact": 0.20,
        "head_position": 0.20,
        "expression": 0.15,
        "stability": 0.10,
    }
}

ACTIVE_SCORING_PROFILE = "code_default"

WEIGHT_POSTURE = SCORING_PROFILES[ACTIVE_SCORING_PROFILE]["posture"]
WEIGHT_GESTURES = SCORING_PROFILES[ACTIVE_SCORING_PROFILE]["gestures"]
WEIGHT_EYE_CONTACT = SCORING_PROFILES[ACTIVE_SCORING_PROFILE]["eye_contact"]
WEIGHT_HEAD_POSITION = SCORING_PROFILES[ACTIVE_SCORING_PROFILE]["head_position"]
WEIGHT_EXPRESSION = SCORING_PROFILES[ACTIVE_SCORING_PROFILE]["expression"]
WEIGHT_STABILITY = SCORING_PROFILES[ACTIVE_SCORING_PROFILE]["stability"]

def set_scoring_profile(profile_name: str) -> None:
    """Dynamically switch between documented scoring profiles."""
    global ACTIVE_SCORING_PROFILE, WEIGHT_POSTURE, WEIGHT_GESTURES, WEIGHT_EYE_CONTACT, WEIGHT_HEAD_POSITION, WEIGHT_EXPRESSION, WEIGHT_STABILITY
    if profile_name in SCORING_PROFILES:
        ACTIVE_SCORING_PROFILE = profile_name
        p = SCORING_PROFILES[profile_name]
        WEIGHT_POSTURE = p["posture"]
        WEIGHT_GESTURES = p["gestures"]
        WEIGHT_EYE_CONTACT = p["eye_contact"]
        WEIGHT_HEAD_POSITION = p["head_position"]
        WEIGHT_EXPRESSION = p["expression"]
        WEIGHT_STABILITY = p["stability"]

# =============================================================================
# 7. VISUALIZATION COLORS & LAYOUT (NOTHING UI: BLACK & WARM BEIGE WITH NEON GREEN MESH)
# =============================================================================
PANEL_W = 390
MARGIN = 14

# Colors (RGBA) — Nothing UI deep black & warm beige palette
PANEL_BG = (10, 10, 12, 245)            # Deep pitch black
PANEL_BORDER = (46, 43, 38, 255)        # Warm subtle graphite-sand border
CARD_BG = (18, 18, 20, 220)             # Matte dark card
CARD_BORDER = (42, 39, 34, 180)         # Warm beige-grey border
TEXT_WHITE = (245, 242, 235, 255)       # Warm beige off-white
TEXT_DIM = (168, 156, 134, 255)         # Warm sand dim text
TEXT_LABEL = (195, 182, 160, 255)       # Elegant letterspaced beige label
TEXT_MUTED = (110, 102, 88, 255)        # Muted sand
BAR_BG = (26, 26, 28, 255)
ACCENT = (232, 222, 200, 255)           # Warm signature beige (#E8DEC8)
ACCENT_DIM = (180, 168, 145, 180)

# Header gradient endpoints (Warm Beige gradient)
HEADER_LEFT = (232, 222, 200, 255)      # Warm Beige
HEADER_RIGHT = (175, 160, 135, 255)     # Deep Sand

# Status Colors — vibrant technical indicators
COLOR_EXCELLENT = (0, 255, 136, 255)    # Neon Emerald Green
COLOR_GOOD = (74, 222, 128, 255)        # Clean Green
COLOR_AVERAGE = (245, 180, 60, 255)     # Warm Ochre
COLOR_POOR = (245, 120, 60, 255)        # Orange
COLOR_VERY_POOR = (235, 60, 60, 255)    # Technical Red
COLOR_UNCERTAIN = (140, 135, 125, 255)  # Sand Slate

# Bar gradient endpoints (high→low)
BAR_HIGH = (0, 255, 136, 255)
BAR_MID = (245, 180, 60, 255)
BAR_LOW = (235, 60, 60, 255)

# Landmark Wireframe Colors — Vibrant Neon Green MediaPipe Mesh
LM_POSE_DOT = (0, 255, 128, 255)        # Neon green pose joints
LM_POSE_LINE = (0, 255, 128, 200)       # Vivid green skeleton lines
LM_FACE_LINE = (0, 255, 128, 180)       # Green face contours
LM_FACE_DOT = (0, 255, 128, 220)        # Green facial landmark dots
LM_IRIS = (0, 255, 220, 255)            # Cyan-green glowing iris center
LM_FACE = (0, 255, 128, 190)            # Vivid green face points
LM_HAND_LINE = (0, 255, 128, 190)       # Green hand joint lines
LM_HAND_DOT = (0, 255, 128, 255)        # Green fingertip dots

# Canvas
CANVAS_BG = (8, 8, 10, 255)
VIDEO_BORDER_RADIUS = 8


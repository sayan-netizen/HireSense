"""
Visualization Module

Creates a premium, real-time glassmorphism dashboard overlay for the
Interview Posture Analyzer using Pillow for anti-aliased drawing and
TrueType fonts.
"""

import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from . import config
except (ImportError, ValueError):
    import config

# ─── Configuration ───────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FONT_REGULAR = os.path.join(MODELS_DIR, "Roboto-Regular.ttf")
FONT_BOLD = os.path.join(MODELS_DIR, "Roboto-Bold.ttf")

# Pose landmark indices used for skeleton drawing
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# ─── Extended Color Palette ─────────────────────────────────────
# Deep glassmorphism backgrounds
CARD_BG_DEEP = (16, 16, 26, 220)
CARD_HIGHLIGHT = (255, 255, 255, 6)       # subtle top-edge highlight
CARD_INNER_SHADOW = (0, 0, 0, 30)         # inner shadow feel

# Glow colors for score bars
GLOW_GREEN = (52, 211, 153, 40)
GLOW_AMBER = (250, 204, 21, 35)
GLOW_RED = (239, 68, 68, 30)

# Status icon colors
ICON_GOOD = (74, 222, 128, 255)
ICON_WARN = (250, 204, 21, 255)
ICON_BAD = (239, 68, 68, 255)
ICON_NEUTRAL = (148, 163, 184, 255)

# Header mesh pattern
HEADER_MESH_DOT = (255, 255, 255, 12)

# Section divider
DIVIDER_COLOR = (60, 60, 80, 100)

# ─── Utility Functions ──────────────────────────────────────────

def _score_color(score: float):
    if score >= 85:
        return config.COLOR_EXCELLENT
    if score >= 70:
        return config.COLOR_GOOD
    if score >= 55:
        return config.COLOR_AVERAGE
    if score >= 40:
        return config.COLOR_POOR
    return config.COLOR_VERY_POOR


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linearly interpolate between two RGBA colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _bar_gradient_color(score: float) -> tuple:
    """Return a smooth gradient color for a score bar (0-100)."""
    if score >= 70:
        return _lerp_color(config.BAR_MID, config.BAR_HIGH, (score - 70) / 30.0)
    elif score >= 40:
        return _lerp_color(config.BAR_LOW, config.BAR_MID, (score - 40) / 30.0)
    else:
        return config.BAR_LOW


def _glow_color_for_score(score: float) -> tuple:
    """Return a soft glow color based on score range."""
    if score >= 70:
        return GLOW_GREEN
    elif score >= 40:
        return GLOW_AMBER
    else:
        return GLOW_RED


def _draw_rounded_rect_shadow(img: Image.Image, rect, radius=12, shadow_offset=4, shadow_blur=8):
    """Draw a subtle drop shadow behind a rounded rectangle."""
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sr = [rect[0] + shadow_offset, rect[1] + shadow_offset,
          rect[2] + shadow_offset, rect[3] + shadow_offset]
    sd.rounded_rectangle(sr, radius=radius, fill=(0, 0, 0, 50))
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    return Image.alpha_composite(img, shadow)


def _draw_card_bg(draw: ImageDraw.ImageDraw, rect, radius=10):
    """Draw a frosted glass card with highlight edge and inner depth."""
    x0, y0, x1, y1 = rect
    # Main card background
    draw.rounded_rectangle(rect, radius=radius, fill=CARD_BG_DEEP,
                           outline=config.CARD_BORDER, width=1)
    # Top edge highlight (simulates light from above)
    draw.line([(x0 + radius, y0 + 1), (x1 - radius, y0 + 1)],
              fill=CARD_HIGHLIGHT, width=1)


class DashboardVisualizer:
    """Draws a premium dashboard and body/face landmarks using Pillow."""

    def __init__(self) -> None:
        try:
            self.font_header = ImageFont.truetype(FONT_BOLD, 18)
            self.font_header_sub = ImageFont.truetype(FONT_REGULAR, 11)
            self.font_title = ImageFont.truetype(FONT_BOLD, 11)
            self.font_section = ImageFont.truetype(FONT_BOLD, 10)
            self.font_score_xl = ImageFont.truetype(FONT_BOLD, 48)
            self.font_score_unit = ImageFont.truetype(FONT_REGULAR, 14)
            self.font_label = ImageFont.truetype(FONT_REGULAR, 12)
            self.font_value = ImageFont.truetype(FONT_BOLD, 12)
            self.font_small = ImageFont.truetype(FONT_REGULAR, 11)
            self.font_tiny = ImageFont.truetype(FONT_REGULAR, 10)
            self.font_bar_num = ImageFont.truetype(FONT_BOLD, 10)
            self.font_event = ImageFont.truetype(FONT_REGULAR, 10)
            self.font_event_ts = ImageFont.truetype(FONT_BOLD, 9)
            self.font_badge = ImageFont.truetype(FONT_BOLD, 10)
            self.font_gauge_label = ImageFont.truetype(FONT_BOLD, 11)
            self.font_calibration_big = ImageFont.truetype(FONT_BOLD, 64)
            self.font_calibration_label = ImageFont.truetype(FONT_BOLD, 18)
            self.font_calibration_sub = ImageFont.truetype(FONT_REGULAR, 13)
            self.font_timer = ImageFont.truetype(FONT_BOLD, 14)
            self.font_fps = ImageFont.truetype(FONT_REGULAR, 9)
        except Exception:
            fallback = ImageFont.load_default()
            for attr in ['font_header', 'font_header_sub', 'font_title', 'font_section',
                         'font_score_xl', 'font_score_unit',
                         'font_label', 'font_value', 'font_small', 'font_tiny',
                         'font_bar_num', 'font_event', 'font_event_ts', 'font_badge',
                         'font_gauge_label', 'font_calibration_big',
                         'font_calibration_label', 'font_calibration_sub',
                         'font_timer', 'font_fps']:
                setattr(self, attr, fallback)

    # ─── Main Render ─────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        scores: dict,
        posture_data: dict,
        face_data: dict,
        hand_data: dict,
        fps: float,
        session_time: float,
        pose_landmarks=None,
        face_landmarks=None,
        hand_landmarks_list=None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        out_w, out_h = 1280, 720

        panel_w = config.PANEL_W
        mg = config.MARGIN

        video_x = mg
        video_y = mg
        video_w = out_w - panel_w - 3 * mg
        video_h = out_h - 2 * mg

        # Scale & crop video frame
        scale = video_h / h
        new_frame_w = int(w * scale)
        scaled_frame = cv2.resize(frame, (new_frame_w, video_h), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
        pil_video = Image.fromarray(rgb_frame).convert("RGBA")

        # Draw landmarks on video
        video_overlay = Image.new("RGBA", (new_frame_w, video_h), (0, 0, 0, 0))
        vdraw = ImageDraw.Draw(video_overlay)

        if pose_landmarks is not None:
            self._draw_pose_skeleton(vdraw, pose_landmarks, new_frame_w, video_h)
        if face_landmarks is not None:
            self._draw_face_points(vdraw, face_landmarks, new_frame_w, video_h)
        if hand_landmarks_list is not None:
            self._draw_hand_points(vdraw, hand_landmarks_list, new_frame_w, video_h)
        if hand_data.get("obscene_gesture", False):
            self._centered_text(vdraw, "[!] OBSCENE GESTURE", new_frame_w, video_h // 2,
                                self.font_calibration_label, config.COLOR_VERY_POOR)

        pil_video = Image.alpha_composite(pil_video, video_overlay)

        # Crop to fit video area
        if new_frame_w > video_w:
            crop_x = (new_frame_w - video_w) // 2
            pil_video = pil_video.crop((crop_x, 0, crop_x + video_w, video_h))
        elif new_frame_w < video_w:
            pad = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 255))
            pad.paste(pil_video, ((video_w - new_frame_w) // 2, 0))
            pil_video = pad

        # --- Build canvas ---
        base = Image.new("RGBA", (out_w, out_h), config.CANVAS_BG)

        # Apply subtle rounded border to the video feed
        video_with_border = self._apply_video_border(pil_video, video_w, video_h)
        base.paste(video_with_border, (video_x, video_y), video_with_border)

        # --- Panel ---
        panel_x = video_x + video_w + mg
        panel_rect = [panel_x, mg, panel_x + panel_w, out_h - mg]

        # Panel shadow (deeper)
        base = _draw_rounded_rect_shadow(base, panel_rect, radius=16, shadow_offset=4, shadow_blur=14)

        draw = ImageDraw.Draw(base)
        draw.rounded_rectangle(panel_rect, radius=16, fill=config.PANEL_BG, outline=config.PANEL_BORDER, width=1)

        px = panel_rect[0] + 18
        py = panel_rect[1]
        inner_w = panel_w - 36

        # ── Header banner with gradient ──
        py = self._draw_header_banner(base, draw, panel_rect, px, py, inner_w, session_time, fps)

        # ── Status Card ──
        py = self._draw_status_card(base, draw, px, py + 6, inner_w, posture_data, face_data, hand_data)

        # ── Score Bars ──
        py = self._draw_scores_card(base, draw, px, py + 6, inner_w, scores)

        # ── Overall Gauge ──
        py = self._draw_overall_gauge(base, draw, px, py + 4, inner_w, scores)

        # ── Event Log ──
        self._draw_event_log(draw, px, py + 4, inner_w, scores.get("recent_events", []),
                             panel_rect[3] - 28)

        # ── Footer ──
        self._draw_footer(draw, px, panel_rect[3] - 24, inner_w)

        return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ─── Calibration Overlay ─────────────────────────────────────

    def draw_calibration(
        self,
        frame: np.ndarray,
        countdown: int,
        person_detected: bool,
        pose_lm=None,
        face_lm=None,
        progress_pct: float = 0.0,
    ) -> np.ndarray:
        """
        Draws live calibration HUD and neon green wireframe landmarks directly on the camera frame.
        Maintains native camera resolution for smooth WebRTC streaming.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb).convert("RGBA")
        draw = ImageDraw.Draw(pil_frame)

        # 1. Draw Neon Green Facial Mesh Wireframe if detected
        if face_lm is not None:
            self._draw_face_points(draw, face_lm, w, h)

        # 2. Draw Neon Green Pose Skeleton if detected
        if pose_lm is not None:
            self._draw_pose_skeleton(draw, pose_lm, w, h)

        # 3. Floating Technical Nothing UI Calibration Banner (Top-Centered)
        banner_w = min(w - 32, 540)
        banner_h = 70
        bx1 = (w - banner_w) // 2
        by1 = 18
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        # Semi-transparent dark pill background with warm sand border
        draw.rounded_rectangle(
            [bx1, by1, bx2, by2],
            radius=6,
            fill=(10, 10, 12, 220),
            outline=config.PANEL_BORDER,
            width=1,
        )

        if person_detected and countdown > 0:
            # Active Calibration Header
            hdr_text = f"● CALIBRATING BASELINE // [ {countdown:02d}s ]"
            draw.text((bx1 + 16, by1 + 12), hdr_text, font=self.font_title, fill=(0, 255, 136, 255))
            pct_text = f"{int(progress_pct)}%"
            draw.text((bx2 - 50, by1 + 12), pct_text, font=self.font_title, fill=config.TEXT_WHITE)

            # Sleek progress bar
            bar_x1 = bx1 + 16
            bar_y1 = by1 + 34
            bar_x2 = bx2 - 16
            bar_y2 = bar_y1 + 4
            draw.rounded_rectangle([bar_x1, bar_y1, bar_x2, bar_y2], radius=2, fill=(30, 30, 34, 255))
            
            fill_pct = max(0.0, min(1.0, progress_pct / 100.0))
            if fill_pct > 0:
                draw.rounded_rectangle(
                    [bar_x1, bar_y1, bar_x1 + int((bar_x2 - bar_x1) * fill_pct), bar_y2],
                    radius=2,
                    fill=(0, 255, 136, 255),
                )

            # Instructions subtext
            draw.text(
                (bx1 + 16, by1 + 46),
                "SIT UPRIGHT · MAINTAIN EYE CONTACT · GREEN MESH VERIFIED",
                font=self.font_small,
                fill=config.TEXT_DIM,
            )
        elif person_detected and countdown == 0:
            draw.text(
                (bx1 + 16, by1 + 16),
                "● BASELINE CAPTURED // READY FOR INTERVIEW",
                font=self.font_title,
                fill=(0, 255, 136, 255),
            )
            draw.text(
                (bx1 + 16, by1 + 42),
                "CLICK 'PROCEED TO LIVE INTERVIEW' TO COMMENCE",
                font=self.font_small,
                fill=config.TEXT_DIM,
            )
        else:
            # Waiting for detection
            draw.text(
                (bx1 + 16, by1 + 16),
                "○ DETECTING CANDIDATE // WAITING FOR FRAME",
                font=self.font_title,
                fill=(245, 180, 60, 255),
            )
            draw.text(
                (bx1 + 16, by1 + 42),
                "POSITION HEAD & SHOULDERS IN CENTER FRAME",
                font=self.font_small,
                fill=config.TEXT_DIM,
            )

        return cv2.cvtColor(np.array(pil_frame.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ─── Dashboard Sections ──────────────────────────────────────

    def _apply_video_border(self, pil_video: Image.Image, vw: int, vh: int) -> Image.Image:
        """Apply a rounded-corner mask and subtle border to the video feed."""
        r = config.VIDEO_BORDER_RADIUS
        mask = Image.new("L", (vw, vh), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([0, 0, vw, vh], radius=r, fill=255)

        bordered = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
        bordered.paste(pil_video, (0, 0), mask)

        # Draw thin border on top
        border_overlay = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(border_overlay)
        bdraw.rounded_rectangle([0, 0, vw - 1, vh - 1], radius=r,
                                outline=config.PANEL_BORDER, width=1)
        bordered = Image.alpha_composite(bordered, border_overlay)
        return bordered

    def draw_mesh_only(
        self,
        frame: np.ndarray,
        pose_lm=None,
        face_lm=None,
        hand_lms_list=None,
    ) -> np.ndarray:
        """
        Draws ONLY the neon green wireframe mesh on face, pose, and hands.
        Direct in-place OpenCV drawing for maximum performance and minimum latency.
        """
        out = frame
        h, w = out.shape[:2]

        GREEN = (128, 255, 0)       # BGR neon green
        CYAN  = (220, 255, 0)       # BGR cyan-green for iris
        DIM_GREEN = (80, 200, 0)    # BGR slightly dimmer for thin lines

        # ── Face mesh (OpenCV vectorized polylines) ──
        if face_lm is not None:
            contours = [
                [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],
                [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33],
                [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362],
                [70,63,105,66,107,55,65,52,53,46],
                [336,296,334,293,300,285,295,282,283,276],
                [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61],
                [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78],
                [168,6,197,195,5,4,1,19,94,2],
                [234,127,93,132],
                [454,356,323,361],
            ]
            for loop in contours:
                pts = []
                for idx in loop:
                    try:
                        p = face_lm[idx]
                        pts.append((int(p.x * w), int(p.y * h)))
                    except (IndexError, AttributeError):
                        pass
                if len(pts) > 1:
                    cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=False, color=DIM_GREEN, thickness=1, lineType=cv2.LINE_AA)

            # Iris
            for idx in (468, 473):
                try:
                    p = face_lm[idx]
                    pt = (int(p.x * w), int(p.y * h))
                    cv2.circle(out, pt, 5, CYAN, 1, cv2.LINE_AA)
                    cv2.circle(out, pt, 2, CYAN, -1, cv2.LINE_AA)
                except (IndexError, AttributeError):
                    pass

            # Key dots
            for idx in (1, 33, 263, 61, 291, 152, 10, 168):
                try:
                    p = face_lm[idx]
                    cv2.circle(out, (int(p.x * w), int(p.y * h)), 2, GREEN, -1, cv2.LINE_AA)
                except (IndexError, AttributeError):
                    pass

        # ── Pose skeleton (OpenCV native) ──
        if pose_lm is not None:
            def ppt(idx):
                p = pose_lm[idx]
                vis = getattr(p, 'visibility', 1.0)
                if vis is not None and vis < 0.5:
                    return None
                return (int(p.x * w), int(p.y * h))

            ls, rs = ppt(LEFT_SHOULDER), ppt(RIGHT_SHOULDER)
            lh, rh = ppt(LEFT_HIP), ppt(RIGHT_HIP)
            nose = ppt(NOSE)
            le, re = ppt(LEFT_ELBOW), ppt(RIGHT_ELBOW)
            lwr, rwr = ppt(LEFT_WRIST), ppt(RIGHT_WRIST)

            for p1, p2 in ((ls,rs),(lh,rh),(ls,lh),(rs,rh),(ls,le),(le,lwr),(rs,re),(re,rwr)):
                if p1 and p2:
                    cv2.line(out, p1, p2, GREEN, 2, cv2.LINE_AA)

            if ls and rs and nose:
                mid_s = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
                cv2.line(out, mid_s, nose, GREEN, 2, cv2.LINE_AA)
                if lh and rh:
                    mid_h = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
                    cv2.line(out, mid_s, mid_h, GREEN, 2, cv2.LINE_AA)

            for p in (ls, rs, lh, rh, nose, le, re, lwr, rwr):
                if p:
                    cv2.circle(out, p, 4, GREEN, -1, cv2.LINE_AA)

        # ── Hand joints (OpenCV native) ──
        if hand_lms_list:
            hand_conns = (
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                (13,17),(0,17),(17,18),(18,19),(19,20)
            )
            for hand_lm in hand_lms_list:
                pts = {}
                for idx, p in enumerate(hand_lm):
                    px, py = int(p.x * w), int(p.y * h)
                    pts[idx] = (px, py)
                    cv2.circle(out, (px, py), 3, GREEN, -1, cv2.LINE_AA)
                for (i, j) in hand_conns:
                    if i in pts and j in pts:
                        cv2.line(out, pts[i], pts[j], GREEN, 2, cv2.LINE_AA)

        return out

    def _draw_header_banner(self, base: Image.Image, draw: ImageDraw.ImageDraw,
                            panel_rect, px, py, inner_w, session_time, fps):
        """Draw a dramatic gradient header banner with mesh dot pattern."""
        banner_h = 62
        bx0, by0 = panel_rect[0] + 1, panel_rect[1] + 1
        bx1 = panel_rect[2] - 1
        by1 = by0 + banner_h

        # Multi-stop gradient fill (violet → blue → teal)
        for x in range(bx0, bx1):
            t = (x - bx0) / max(1, bx1 - bx0)
            if t < 0.6:
                c = _lerp_color(config.HEADER_LEFT, config.HEADER_RIGHT, t / 0.6)
            else:
                teal = (20, 184, 166, 255)
                c = _lerp_color(config.HEADER_RIGHT, teal, (t - 0.6) / 0.4)
            draw.line([(x, by0), (x, by1)], fill=c)

        # Mesh dot pattern overlay
        for mx in range(bx0 + 8, bx1, 16):
            for my in range(by0 + 8, by1, 16):
                draw.ellipse([mx - 1, my - 1, mx + 1, my + 1], fill=HEADER_MESH_DOT)

        # Round top corners by redrawing panel border on top
        draw.rounded_rectangle(panel_rect, radius=16, outline=config.PANEL_BORDER, width=1)

        # Title text with shadow
        draw.text((px + 1, by0 + 11), "INTERVIEW POSTURE", font=self.font_header,
                  fill=(0, 0, 0, 60))
        draw.text((px, by0 + 10), "INTERVIEW POSTURE", font=self.font_header,
                  fill=(255, 255, 255, 255))
        draw.text((px, by0 + 32), "& CONFIDENCE ANALYZER", font=self.font_header_sub,
                  fill=(220, 220, 240, 200))

        # Session timer (right-aligned) — larger, pill-shaped background
        mins, secs = divmod(int(session_time), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        fps_str = f"{fps:.0f} FPS"

        tbbox = draw.textbbox((0, 0), time_str, font=self.font_timer)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]

        # Timer pill
        pill_x = bx1 - tw - 28
        pill_y = by0 + 8
        pill_w = tw + 20
        pill_h = th + 10
        draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
                               radius=pill_h // 2, fill=(0, 0, 0, 80))
        draw.text((pill_x + 10, pill_y + 5), time_str, font=self.font_timer,
                  fill=(255, 255, 255, 240))

        # FPS label below timer pill
        fbbox = draw.textbbox((0, 0), fps_str, font=self.font_fps)
        fw = fbbox[2] - fbbox[0]
        draw.text((bx1 - fw - 18, by0 + pill_h + 16), fps_str, font=self.font_fps,
                  fill=(200, 200, 220, 160))

        return by1 + 2

    def _draw_section_title(self, draw, px, py, title):
        """Draw a section title with accent dot and subtle underline."""
        # Accent dot
        dot_r = 3
        dot_y = py + 6
        draw.ellipse([px, dot_y - dot_r, px + dot_r * 2, dot_y + dot_r],
                     fill=config.ACCENT)
        # Title text
        draw.text((px + dot_r * 2 + 6, py), title, font=self.font_section,
                  fill=config.TEXT_DIM)
        # Subtle line
        bbox = draw.textbbox((0, 0), title, font=self.font_section)
        tw = bbox[2] - bbox[0]
        line_x = px + dot_r * 2 + 6 + tw + 6
        draw.line([(line_x, dot_y), (px + 340 - 18, dot_y)],
                  fill=DIVIDER_COLOR, width=1)
        return py + 18

    def _draw_separator(self, draw, px, py, w) -> int:
        draw.line([(px, py), (px + w, py)], fill=config.CARD_BORDER, width=1)
        return py + 1

    def _draw_status_card(self, base, draw, px, py, inner_w, posture_data, face_data, hand_data):
        """Draw status labels inside a frosted glass card with pill-shaped values."""
        py = self._draw_section_title(draw, px, py, "LIVE STATUS")

        # Card background
        card_pad = 8
        card_x0 = px - card_pad
        card_y0 = py
        # We'll compute the card height after laying out items

        posture_insuf = posture_data.get("insufficient_data", False)
        face_insuf = face_data.get("insufficient_data", False)
        hand_insuf = hand_data.get("insufficient_data", False)

        def pv(k): return "—" if posture_insuf else posture_data.get(k, "N/A")
        def fv(k): return "—" if face_insuf else face_data.get(k, "N/A")
        def hv(k): return "—" if hand_insuf else hand_data.get(k, "N/A")

        shoulder_val = pv("shoulder_status") if pv("shoulder_status") == "Tense / Elevated" else pv("shoulder_alignment")
        blink_val = "—" if face_insuf else f"{face_data.get('blink_rate', 0):.0f} /min"

        items = [
            ("◈  Posture", pv("posture")),
            ("◈  Shoulders", shoulder_val),
            ("◈  Head", fv("head_position")),
            ("◈  Eye Contact", "—" if face_insuf else f"{face_data.get('eye_contact_pct', 0):.0f}%"),
            ("◈  Blink Rate", blink_val),
            ("◈  Expression", fv("expression")),
            ("◈  Leaning", pv("leaning")),
            ("◈  Gestures", hv("status")),
        ]

        row_h = 21
        card_h = len(items) * row_h + card_pad * 2

        # Draw card background
        _draw_card_bg(draw, [card_x0, card_y0, card_x0 + inner_w + card_pad * 2,
                             card_y0 + card_h], radius=8)

        item_y = card_y0 + card_pad

        for label, value in items:
            # Label
            draw.text((px + 2, item_y), label, font=self.font_small, fill=config.TEXT_LABEL)

            # Value as a colored pill badge
            val_str = str(value)
            color = self._value_color(val_str)
            vbbox = draw.textbbox((0, 0), val_str, font=self.font_badge)
            vw = vbbox[2] - vbbox[0]
            vh = vbbox[3] - vbbox[1]

            pill_pad_x = 8
            pill_pad_y = 2
            pill_x = px + inner_w - vw - pill_pad_x * 2
            pill_y = item_y - 1

            # Pill background (very subtle)
            pill_bg = (color[0], color[1], color[2], 20)
            draw.rounded_rectangle(
                [pill_x, pill_y, pill_x + vw + pill_pad_x * 2, pill_y + vh + pill_pad_y * 2 + 4],
                radius=6, fill=pill_bg
            )
            draw.text((pill_x + pill_pad_x, pill_y + pill_pad_y + 1), val_str,
                      font=self.font_badge, fill=color)

            item_y += row_h

        return card_y0 + card_h + 2

    def _draw_scores_card(self, base, draw, px, py, inner_w, scores):
        """Draw score bars with gradient fills and luminous glow effect."""
        py = self._draw_section_title(draw, px, py, "SCORE BREAKDOWN")

        items = [
            ("Posture", scores.get("posture", 0)),
            ("Eye Contact", scores.get("eye_contact", 0)),
            ("Head Pos.", scores.get("head_position", 0)),
            ("Expression", scores.get("expression", 0)),
            ("Stability", scores.get("stability", 0)),
            ("Gestures", scores.get("gestures", 0)),
        ]

        label_w = 80
        bar_x = px + label_w
        num_w = 34
        bar_w = inner_w - label_w - num_w
        bar_h = 8
        row_h = 24

        # Card background
        card_pad = 8
        card_x0 = px - card_pad
        card_y0 = py
        card_h = len(items) * row_h + card_pad * 2
        _draw_card_bg(draw, [card_x0, card_y0, card_x0 + inner_w + card_pad * 2,
                             card_y0 + card_h], radius=8)

        item_y = card_y0 + card_pad

        for label, score in items:
            # Label
            draw.text((px + 2, item_y - 1), label, font=self.font_small, fill=config.TEXT_LABEL)

            # Bar track with subtle rounded corners
            by = item_y + 2
            draw.rounded_rectangle([bar_x, by, bar_x + bar_w, by + bar_h],
                                   radius=4, fill=config.BAR_BG)

            # Gradient fill
            fill_w = int(bar_w * min(float(score), 100.0) / 100.0)
            if fill_w > 2:
                bar_color = _bar_gradient_color(score)
                draw.rounded_rectangle([bar_x, by, bar_x + fill_w, by + bar_h],
                                       radius=4, fill=bar_color)

                # Glow line underneath the bar (1px lower, wider, transparent)
                glow = _glow_color_for_score(score)
                draw.rounded_rectangle([bar_x + 2, by + bar_h, bar_x + fill_w - 2, by + bar_h + 2],
                                       radius=1, fill=glow)

                # Bright tip dot at end of fill
                tip_x = bar_x + fill_w - 2
                tip_cy = by + bar_h // 2
                draw.ellipse([tip_x - 2, tip_cy - 2, tip_x + 2, tip_cy + 2],
                             fill=(255, 255, 255, 120))

            # Numeric value (right-aligned)
            num_str = f"{score:.0f}"
            nbbox = draw.textbbox((0, 0), num_str, font=self.font_bar_num)
            nw = nbbox[2] - nbbox[0]
            num_color = _bar_gradient_color(score)
            draw.text((px + inner_w - nw, item_y - 1), num_str,
                      font=self.font_bar_num, fill=num_color)

            item_y += row_h

        return card_y0 + card_h + 2

    def _draw_overall_gauge(self, base, draw, px, py, inner_w, scores):
        """Draw a larger arc gauge with double-ring glow effect."""
        overall = scores.get("overall", 0)
        status = scores.get("status", "N/A")
        reliability = scores.get("reliability", 100.0)
        color = _score_color(overall)

        py = self._draw_section_title(draw, px, py, "OVERALL READINESS")

        # Arc gauge — larger and more prominent
        gauge_size = 100
        cx = px + inner_w // 2
        cy = py + gauge_size // 2 + 10

        # Outer glow ring (very faint, larger)
        glow_bbox = [cx - gauge_size // 2 - 6, cy - gauge_size // 2 - 6,
                     cx + gauge_size // 2 + 6, cy + gauge_size // 2 + 6]
        glow_color = (color[0], color[1], color[2], 20)
        draw.arc(glow_bbox, start=160, end=380, fill=glow_color, width=14)

        # Background arc (220° sweep, from 160° to 380°)
        arc_bbox = [cx - gauge_size // 2, cy - gauge_size // 2,
                    cx + gauge_size // 2, cy + gauge_size // 2]
        draw.arc(arc_bbox, start=160, end=380, fill=config.BAR_BG, width=8)

        # Filled arc
        sweep = int(220 * min(overall, 100) / 100.0)
        if sweep > 0:
            draw.arc(arc_bbox, start=160, end=160 + sweep, fill=color, width=8)

            # Inner bright arc (thinner, brighter — creates depth)
            inner_arc_bbox = [cx - gauge_size // 2 + 2, cy - gauge_size // 2 + 2,
                              cx + gauge_size // 2 - 2, cy + gauge_size // 2 - 2]
            bright_color = (min(color[0] + 40, 255), min(color[1] + 40, 255),
                            min(color[2] + 40, 255), 80)
            draw.arc(inner_arc_bbox, start=160, end=160 + sweep, fill=bright_color, width=2)

        # Score text centered in arc
        score_str = f"{overall:.0f}"
        sbbox = draw.textbbox((0, 0), score_str, font=self.font_score_xl)
        sw = sbbox[2] - sbbox[0]
        sh = sbbox[3] - sbbox[1]
        # Text shadow
        draw.text((cx - sw // 2 + 1, cy - sh // 2 - 5), score_str,
                  font=self.font_score_xl, fill=(0, 0, 0, 50))
        draw.text((cx - sw // 2, cy - sh // 2 - 6), score_str,
                  font=self.font_score_xl, fill=color)

        # Unit label
        ubbox = draw.textbbox((0, 0), "/100", font=self.font_tiny)
        uw = ubbox[2] - ubbox[0]
        draw.text((cx - uw // 2, cy + sh // 2 - 8), "/100",
                  font=self.font_tiny, fill=config.TEXT_DIM)

        py = cy + gauge_size // 2 + 14

        # Status badge — larger, with icon
        status_text = status.upper()
        status_icon = self._status_icon(status_text)
        full_status = f"{status_icon}  {status_text}"
        stbbox = draw.textbbox((0, 0), full_status, font=self.font_gauge_label)
        stw = stbbox[2] - stbbox[0]
        badge_x = cx - stw // 2 - 14
        badge_w = stw + 28
        badge_h = 24
        # Badge background with status color at low opacity
        badge_color = (color[0], color[1], color[2], 30)
        draw.rounded_rectangle([badge_x, py, badge_x + badge_w, py + badge_h],
                               radius=badge_h // 2, fill=badge_color,
                               outline=(color[0], color[1], color[2], 100), width=1)
        draw.text((badge_x + 14, py + 5), full_status, font=self.font_gauge_label, fill=color)
        py += badge_h + 8

        # Reliability bar (mini progress bar instead of just text)
        draw.text((px, py), "Tracking Reliability", font=self.font_small, fill=config.TEXT_DIM)

        rel_color = config.COLOR_GOOD if reliability >= 80 else (
            config.COLOR_AVERAGE if reliability >= 50 else config.COLOR_VERY_POOR)

        # Mini reliability bar
        rel_bar_w = 60
        rel_bar_h = 5
        rel_bar_x = px + inner_w - rel_bar_w - 30
        rel_bar_y = py + 5
        draw.rounded_rectangle([rel_bar_x, rel_bar_y, rel_bar_x + rel_bar_w, rel_bar_y + rel_bar_h],
                               radius=2, fill=config.BAR_BG)
        rel_fill = int(rel_bar_w * min(reliability, 100.0) / 100.0)
        if rel_fill > 1:
            draw.rounded_rectangle([rel_bar_x, rel_bar_y, rel_bar_x + rel_fill, rel_bar_y + rel_bar_h],
                                   radius=2, fill=rel_color)

        # Reliability percentage
        rel_str = f"{reliability:.0f}%"
        rbbox = draw.textbbox((0, 0), rel_str, font=self.font_bar_num)
        rw = rbbox[2] - rbbox[0]
        draw.text((px + inner_w - rw, py + 1), rel_str, font=self.font_bar_num, fill=rel_color)

        return py + 18

    def _draw_event_log(self, draw, px, py, inner_w, recent_events, max_y):
        """Draw the event log section with color-coded dot indicators."""
        py = self._draw_section_title(draw, px, py, "EVENT LOG")

        if not recent_events:
            draw.text((px + 8, py), "No events recorded yet", font=self.font_event,
                      fill=config.TEXT_MUTED)
            return py + 16

        for event in reversed(recent_events):
            if py + 16 > max_y:
                break

            # Color-coded dot based on event content
            event_lower = event.lower()
            if "corrected" in event_lower or "good" in event_lower:
                dot_color = ICON_GOOD
            elif "warning" in event_lower or "slouch" in event_lower or "tension" in event_lower:
                dot_color = ICON_WARN
            elif "poor" in event_lower or "excessive" in event_lower:
                dot_color = ICON_BAD
            else:
                dot_color = ICON_NEUTRAL

            # Dot
            dot_cy = py + 6
            draw.ellipse([px + 2, dot_cy - 2, px + 6, dot_cy + 2], fill=dot_color)

            # Split timestamp and event text
            if " — " in event:
                ts, msg = event.split(" — ", 1)
                draw.text((px + 12, py), ts, font=self.font_event_ts, fill=config.ACCENT_DIM)
                draw.text((px + 48, py), msg, font=self.font_event, fill=config.TEXT_WHITE)
            else:
                draw.text((px + 12, py), event, font=self.font_event, fill=config.TEXT_WHITE)
            py += 16

        return py + 4

    def _draw_footer(self, draw, px, py, inner_w):
        """Draw a polished footer with accent divider."""
        # Divider line
        draw.line([(px, py - 4), (px + inner_w, py - 4)], fill=DIVIDER_COLOR, width=1)
        # Footer text
        draw.text((px, py), "Press Q or ESC to quit", font=self.font_tiny, fill=config.TEXT_MUTED)
        # Accent dot at end
        bbox = draw.textbbox((0, 0), "Press Q or ESC to quit", font=self.font_tiny)
        tw = bbox[2] - bbox[0]
        draw.ellipse([px + tw + 6, py + 3, px + tw + 10, py + 7], fill=config.ACCENT_DIM)

    # ─── Landmark Drawing ────────────────────────────────────────

    @staticmethod
    def _draw_pose_skeleton(draw: ImageDraw.ImageDraw, landmarks, w: int, h: int) -> None:
        def pt(idx):
            p = landmarks[idx]
            vis = getattr(p, 'visibility', 1.0)
            if vis is not None and vis < 0.5:
                return None
            return (int(p.x * w), int(p.y * h))

        ls = pt(LEFT_SHOULDER)
        rs = pt(RIGHT_SHOULDER)
        lh = pt(LEFT_HIP)
        rh = pt(RIGHT_HIP)
        nose = pt(NOSE)
        le = pt(LEFT_ELBOW)
        re = pt(RIGHT_ELBOW)
        lw = pt(LEFT_WRIST)
        rw = pt(RIGHT_WRIST)

        # Lines
        line_color = config.LM_POSE_LINE
        dot_color = config.LM_POSE_DOT
        line_w = 2

        connections = [
            (ls, rs), (lh, rh), (ls, lh), (rs, rh),
            (ls, le), (le, lw), (rs, re), (re, rw),
        ]
        for p1, p2 in connections:
            if p1 and p2:
                draw.line([p1, p2], fill=line_color, width=line_w)

        # Neck and spine line
        if ls and rs and nose:
            mid_s = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            draw.line([mid_s, nose], fill=line_color, width=line_w)
            if lh and rh:
                mid_h = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
                draw.line([mid_s, mid_h], fill=line_color, width=line_w)

        # Dots with glow
        all_pts = [ls, rs, lh, rh, nose, le, re, lw, rw]
        for p in all_pts:
            if p:
                # Outer glow
                draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6],
                             fill=(dot_color[0], dot_color[1], dot_color[2], 40))
                # Inner dot
                draw.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=dot_color)

    @staticmethod
    def _draw_face_points(draw: ImageDraw.ImageDraw, landmarks, w: int, h: int) -> None:
        if not landmarks:
            return

        def get_pt(idx):
            try:
                p = landmarks[idx]
                return (int(p.x * w), int(p.y * h))
            except (IndexError, AttributeError):
                return None

        face_line = getattr(config, "LM_FACE_LINE", (0, 255, 128, 190))
        face_dot = getattr(config, "LM_FACE_DOT", (0, 255, 128, 220))
        iris_color = getattr(config, "LM_IRIS", (0, 255, 220, 255))

        # Vibrant green facial wireframe contour loops
        contours = [
            # Jawline contour
            [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
            # Left Eye
            [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
            # Right Eye
            [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
            # Left Eyebrow
            [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            # Right Eyebrow
            [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
            # Outer Lips
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
            # Inner Lips
            [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
            # Nose bridge & tip
            [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
            # Cheek cross lines
            [234, 127, 93, 132],
            [454, 356, 323, 361],
        ]

        # Draw green contour wireframe lines
        for loop in contours:
            for k in range(len(loop) - 1):
                p1 = get_pt(loop[k])
                p2 = get_pt(loop[k + 1])
                if p1 and p2:
                    draw.line([p1, p2], fill=face_line, width=1)

        # Iris rings & centers with glowing cyan-green
        for idx in [468, 473]:
            pt = get_pt(idx)
            if pt:
                x, y = pt
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(0, 255, 220, 60))
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=iris_color)

        # Key landmark points
        key_dots = [1, 33, 263, 61, 291, 152, 10, 168]
        for idx in key_dots:
            pt = get_pt(idx)
            if pt:
                x, y = pt
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=face_dot)

    @staticmethod
    def _draw_hand_points(draw: ImageDraw.ImageDraw, hand_landmarks_list, w: int, h: int) -> None:
        if not hand_landmarks_list:
            return

        connections = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (0,17), (17,18), (18,19), (19,20)
        ]

        hand_line = getattr(config, "LM_HAND_LINE", (0, 255, 128, 200))
        hand_dot = getattr(config, "LM_HAND_DOT", (0, 255, 128, 255))

        for hand_lm in hand_landmarks_list:
            for (i, j) in connections:
                try:
                    p1, p2 = hand_lm[i], hand_lm[j]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    draw.line([(x1, y1), (x2, y2)], fill=hand_line, width=2)
                except IndexError:
                    pass
            for p in hand_lm:
                px, py = int(p.x * w), int(p.y * h)
                draw.ellipse([px-3, py-3, px+3, py+3], fill=hand_dot)

    # ─── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _centered_text(draw, text, img_w, y, font, color):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        x = (img_w - text_w) // 2
        draw.text((x, y), text, font=font, fill=color)

    @staticmethod
    def _status_icon(status_text: str) -> str:
        """Return a technical indicator for the overall status."""
        s = status_text.lower()
        if "excellent" in s:
            return "[+]"
        if "good" in s:
            return "[*]"
        if "average" in s:
            return "[=]"
        if "needs" in s or "poor" in s:
            return "[-]"
        return "[?]"

    @staticmethod
    def _value_color(value: str):
        vl = value.lower()

        positive = ("excellent", "good", "balanced", "centered", "positive", "none",
                     "resting", "engaged", "natural gesture", "direct", "normal movement", "relaxed")
        if any(w in vl for w in positive):
            return config.COLOR_GOOD

        moderate = ("slight", "fair", "neutral", "occasional distraction", "tilted",
                     "slight slouch", "one hand hidden", "near-camera")
        if any(w in vl for w in moderate):
            return config.COLOR_AVERAGE

        negative = ("poor", "significant", "excessive", "frequently", "tense",
                     "sustained away", "moderate", "slouching", "possible fidgeting",
                     "touching face", "obscene", "forward head", "elevated", "shrugged")
        if any(w in vl for w in negative):
            return config.COLOR_POOR

        insufficient = ("insufficient", "unknown", "hands not visible", "—")
        if any(w in vl for w in insufficient):
            return config.COLOR_UNCERTAIN

        if "%" in vl:
            try:
                pct = float(vl.replace("%", ""))
                if pct >= 70:
                    return config.COLOR_GOOD
                if pct >= 45:
                    return config.COLOR_AVERAGE
                return config.COLOR_POOR
            except ValueError:
                pass

        return config.TEXT_WHITE

#!/usr/bin/env python3
"""Generate the demonstration documents the problem statement names:
a P&ID-style engineering drawing and a handwritten field note.

These are synthetic, drawn here from scratch, so nothing proprietary is
included -- the PS says open/sample documents are expected for the demo.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parent.parent / "workspace"
WS.mkdir(parents=True, exist_ok=True)


def font(name, size):
    for cand in (name, "arial.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def speckle(img, n=2500, lo=185, hi=225):
    w, h = img.size
    for _ in range(n):
        img.putpixel((random.randint(0, w - 1), random.randint(0, h - 1)),
                     (random.randint(lo, hi),) * 3)


# ── 1. P&ID-style engineering drawing ───────────────────────────────────
def make_pid():
    W, H = 1400, 950
    img = Image.new("RGB", (W, H), (252, 252, 248))
    d = ImageDraw.Draw(img)
    f_t = font("arialbd.ttf", 30)
    f_l = font("arial.ttf", 17)
    f_s = font("arial.ttf", 14)
    ink = (25, 25, 30)

    # title block
    d.rectangle([20, 20, W - 20, H - 20], outline=ink, width=2)
    d.rectangle([20, 20, W - 20, 90], outline=ink, width=2)
    d.text((40, 38), "P&ID - CRUDE TRANSFER LOOP / CDU-2", font=f_t, fill=ink)
    d.text((W - 430, 32), "DRG NO: MRPL-PID-CDU2-014", font=f_s, fill=ink)
    d.text((W - 430, 54), "REV: 3   DATE: 04-SEP-2026", font=f_s, fill=ink)
    d.text((W - 430, 72), "SCALE: NTS   SHEET 1 OF 1", font=f_s, fill=ink)

    # storage tank
    d.rectangle([70, 250, 250, 470], outline=ink, width=3)
    d.ellipse([70, 232, 250, 268], outline=ink, width=3)
    d.text((104, 490), "TK-101", font=f_l, fill=ink)
    d.text((80, 512), "CRUDE STORAGE", font=f_s, fill=ink)
    d.text((96, 340), "LI\n101", font=f_s, fill=ink)

    # suction line
    d.line([250, 380, 430, 380], fill=ink, width=3)
    # gate valve (bowtie)
    d.polygon([(330, 366), (330, 394), (358, 366), (358, 394)], outline=ink, width=2)
    d.text((326, 400), "V-101", font=f_s, fill=ink)

    # pump P-201
    d.ellipse([430, 340, 510, 420], outline=ink, width=3)
    d.polygon([(470, 340), (510, 380), (470, 420)], outline=ink, width=2)
    d.text((432, 430), "P-201", font=f_l, fill=ink)
    d.text((408, 452), "CENTRIFUGAL PUMP", font=f_s, fill=ink)

    # instrument bubbles on the pump
    for cx, cy, tag in ((470, 285, "VI\n201"), (555, 285, "TI\n201"),
                        (640, 285, "PI\n202")):
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], outline=ink, width=2)
        d.line([cx - 26, cy, cx + 26, cy], fill=ink, width=1)
        d.text((cx - 12, cy - 20), tag, font=f_s, fill=ink)
    d.line([470, 340, 470, 311], fill=ink, width=1)
    d.line([555, 311, 555, 380], fill=ink, width=1)
    d.line([640, 311, 640, 380], fill=ink, width=1)

    # discharge line with check valve and control valve
    d.line([510, 380, 900, 380], fill=ink, width=3)
    d.polygon([(690, 366), (690, 394), (718, 380)], outline=ink, width=2)
    d.text((684, 400), "NRV-201", font=f_s, fill=ink)

    d.polygon([(820, 366), (820, 394), (848, 366), (848, 394)], outline=ink, width=2)
    d.line([834, 366, 834, 336], fill=ink, width=1)
    d.polygon([(818, 316), (850, 316), (834, 336)], outline=ink, width=2)
    d.text((806, 400), "FCV-203", font=f_s, fill=ink)

    # heat exchanger
    d.rectangle([900, 320, 1080, 440], outline=ink, width=3)
    for x in range(920, 1070, 30):
        d.line([x, 330, x, 430], fill=ink, width=1)
    d.text((940, 452), "E-301", font=f_l, fill=ink)
    d.text((906, 474), "FEED PREHEATER", font=f_s, fill=ink)

    # to column
    d.line([1080, 380, 1310, 380], fill=ink, width=3)
    d.polygon([(1290, 372), (1310, 380), (1290, 388)], fill=ink)
    d.text((1150, 350), "TO COLUMN C-401", font=f_s, fill=ink)

    # recycle line (dashed)
    for x in range(560, 900, 22):
        d.line([x, 600, x + 12, 600], fill=ink, width=2)
    d.line([880, 380, 880, 600], fill=ink, width=1)
    d.line([560, 600, 560, 380], fill=ink, width=1)
    d.text((660, 612), "MIN-FLOW RECYCLE", font=f_s, fill=ink)

    # notes
    d.text((70, 700), "NOTES:", font=f_l, fill=ink)
    for i, n in enumerate([
        "1. P-201 SEAL FLUSH PER SOP-MECH-041; VIBRATION ALERT LIMIT 4.5 mm/s RMS.",
        "2. NRV-201 TO BE PROVEN BEFORE RESTART AFTER ANY SEAL REPLACEMENT.",
        "3. FCV-203 FAIL POSITION: FAIL CLOSED.",
        "4. LINE SIZE 6 IN SCH 40; DESIGN PRESSURE 19 barg; DESIGN TEMP 120 C.",
    ]):
        d.text((70, 730 + i * 26), n, font=f_s, fill=ink)

    speckle(img, 2200)
    out = WS / "pid_crude_transfer.png"
    img.save(out)
    print("wrote", out.name)


# ── 2. Handwritten field note ───────────────────────────────────────────
def make_handwritten():
    W, H = 1000, 1350
    img = Image.new("RGB", (W, H), (250, 247, 232))
    d = ImageDraw.Draw(img)
    # ruled lines
    for y in range(150, H - 60, 58):
        d.line([60, y, W - 60, y], fill=(205, 212, 222), width=1)
    d.line([110, 90, 110, H - 60], fill=(226, 180, 180), width=2)

    hand = font("Inkfree.ttf", 40)
    hand_s = font("Inkfree.ttf", 34)

    d.text((130, 70), "Shift Log - Night", font=hand, fill=(28, 34, 92))
    lines = [
        "12-Sep-2026  22:40 hrs",
        "Pump P-201 running, flow steady.",
        "Noticed slight oil seepage at",
        "NDE bearing housing - approx",
        "3 drops / min. Wiped + tightened",
        "gland nut by 1/4 turn.",
        "Vibration re-checked = 2.8 mm/s",
        "(limit 4.5) - within range.",
        "Bearing temp 71 C, stable.",
        "Recommend: monitor each shift",
        "for 3 days. Inform day mech.",
        "",
        "- R. Kamath, Fitter Gr-I",
    ]
    y = 165
    for ln in lines:
        jitter = random.randint(-2, 2)
        d.text((132 + jitter, y), ln, font=hand_s, fill=(30, 38, 105))
        y += 58

    speckle(img, 3000, 200, 235)
    out = WS / "handwritten_shift_note.png"
    img.save(out)
    print("wrote", out.name)


if __name__ == "__main__":
    random.seed(7)
    make_pid()
    make_handwritten()

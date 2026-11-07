"""Build the HELIX wordmark and icon as SVG, light and dark.

Letters come from Archivo Black (SIL OFL 1.1, fonts/OFL.txt), converted to
outlines so the SVGs need no font. The X is two helix strands, cyan top-left
to bottom-right and magenta top-right to bottom-left, magenta passing over cyan
at the crossing, with "<>" and "{}" as the rungs. The X is as wide as the H.

    uv run --no-project --with fonttools==4.60.1 --with numpy==2.3.4 \\
        python docs/assets/logo/make_logo.py [OUT_DIR] [--debug]

--debug prints where each rung landed against the centre of its opening.
"""
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont
import numpy as np

HERE = Path(__file__).parent
DEBUG = '--debug' in sys.argv
font = TTFont(HERE / "fonts" / "ArchivoBlack.ttf")
gs = font.getGlyphSet()
cmap = font.getBestCmap()
CAP = font["OS/2"].sCapHeight

PALETTES = {
    "light": dict(text="#1f2328", cyan="#1497b3", magenta="#b8269f", purple="#6f3aa0"),
    "dark": dict(text="#e6edf3", cyan="#3fc1da", magenta="#e25ad8", purple="#a982e8"),
}


def glyph(ch):
    name = cmap[ord(ch)]
    pen = SVGPathPen(gs)
    gs[name].draw(pen)
    bp = BoundsPen(gs)
    gs[name].draw(bp)
    return pen.getCommands(), gs[name].width, bp.bounds


def bez(p0, p1, p2, p3, t):
    u = 1 - t
    return tuple(u**3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t**3 * d for a, b, c, d in zip(p0, p1, p2, p3))


def helix_x(x0, top, w, h, stroke, c, rungs, uid, margin=0.06):
    """Two crossing strands in a w x h box at (x0, top). Returns SVG markup."""
    inset = stroke / 2
    left, right, bot = x0 + inset, x0 + w - inset, top + h
    k = 0.55  # how far the vertical tangents reach: higher = more S-shaped
    cyan = ((left, top), (left, top + k * h), (right, bot - k * h), (right, bot))
    mag = ((right, top), (right, top + k * h), (left, bot - k * h), (left, bot))

    def d(cv):
        (a, b, cc, e) = cv
        return f"M{a[0]:.1f} {a[1]:.1f} C{b[0]:.1f} {b[1]:.1f} {cc[0]:.1f} {cc[1]:.1f} {e[0]:.1f} {e[1]:.1f}"

    out = []
    for pair in rungs:  # rungs sit under the strands
        out.append(place_pair(pair, cyan, mag, x0, top, w, h, stroke, c["purple"], margin * h))
    gap = stroke * 0.32
    out.append(
        f'<mask id="m{uid}" maskUnits="userSpaceOnUse"><rect x="{x0 - stroke}" y="{top - stroke}" width="{w + 2 * stroke}" height="{h + 2 * stroke}" fill="#fff"/>'
        f'<path d="{d(mag)}" stroke="#000" stroke-width="{stroke + 2 * gap:.1f}" fill="none"/></mask>'
    )
    out.append(f'<path d="{d(cyan)}" stroke="{c["cyan"]}" stroke-width="{stroke:.1f}" fill="none" mask="url(#m{uid})"/>')
    out.append(f'<path d="{d(mag)}" stroke="{c["magenta"]}" stroke-width="{stroke:.1f}" fill="none"/>')
    return "\n".join(out)


def place_pair(pair, cyan, mag, x0, top, w, h, stroke, color, margin):
    """Fit a glyph pair (e.g. "<>") in the opening above or below the crossing.

    The pair sits on the X's vertical axis. Vertically, the space between it
    and the cap line (or baseline) equals the space between it and the strands
    on the crossing side, measured straight down (or up) from every column of
    the pair, so the nearest point of the narrowing V counts. The pair is the
    largest for which that holds while it keeps `margin` from the strands in
    every direction.
    """
    which, chars = pair
    gl = []
    for ch in chars:
        cmd, adv, (gx0, gy0, gx1, gy1) = glyph(ch)
        gl.append((cmd, gx0, gy0, gx1, gy1))
    ink_h = max(g[4] - g[2] for g in gl)
    sp = 0.18  # gap between the two glyphs, as a fraction of height
    aspect = (sum(g[3] - g[1] for g in gl) / ink_h) + sp

    ts = np.linspace(0, 1, 1500)
    pts = np.concatenate([np.array([bez(*cv, t) for t in ts]) for cv in (cyan, mag)])
    cx = x0 + w / 2
    bottom = top + h
    mid = top + h / 2

    # Free-space grid over the X: True where no strand covers the point.
    xs_ = np.linspace(x0, x0 + w, 600)
    ys_ = np.linspace(top, bottom, 1200)
    free = np.ones((len(ys_), len(xs_)), bool)
    for r, y in enumerate(ys_):
        near = pts[np.abs(pts[:, 1] - y) <= stroke / 2 + 1]
        if len(near):
            d = np.hypot(xs_[:, None] - near[None, :, 0], y - near[None, :, 1]).min(1)
            free[r] = d > stroke / 2

    # For every row and column: the nearest strand row at or below it
    # (n_rows if none) and the nearest one above it (-1 if none), so each gap
    # query below is a lookup instead of a scan.
    n_rows = len(ys_)
    blocked = ~free
    next_blocked = np.full(blocked.shape, n_rows)
    prev_blocked = np.full(blocked.shape, -1)
    for r in range(n_rows - 1, -1, -1):
        below = next_blocked[r + 1] if r + 1 < n_rows else np.full(len(xs_), n_rows)
        next_blocked[r] = np.where(blocked[r], r, below)
    for r in range(n_rows):
        above = prev_blocked[r - 1] if r > 0 else np.full(len(xs_), -1)
        prev_blocked[r] = np.where(blocked[r], r, above)
    ys_ext = np.append(ys_, np.nan)  # index n_rows = no strand

    def strand_clearance(bh, yc):
        bw = aspect * bh
        bx0, bx1, by0, by1 = cx - bw / 2, cx + bw / 2, yc - bh / 2, yc + bh / 2
        dx = np.maximum.reduce([bx0 - pts[:, 0], np.zeros(len(pts)), pts[:, 0] - bx1])
        dy = np.maximum.reduce([by0 - pts[:, 1], np.zeros(len(pts)), pts[:, 1] - by1])
        return np.hypot(dx, dy).min() - stroke / 2

    def gaps(bh, yc):
        """(gap to the cap line or baseline, gap to the strands toward the crossing)."""
        bw = aspect * bh
        by0, by1 = yc - bh / 2, yc + bh / 2
        cols = np.where((xs_ >= cx - bw / 2) & (xs_ <= cx + bw / 2))[0]
        if which == "top":
            r0 = np.searchsorted(ys_, by1)
            nb = next_blocked[r0, cols] if r0 < n_rows else np.full(len(cols), n_rows)
            far = np.where(nb < n_rows, ys_ext[nb] - by1, h).min()
            return by0 - top, float(far)
        r1 = np.searchsorted(ys_, by0)
        pb = prev_blocked[r1 - 1, cols] if r1 > 0 else np.full(len(cols), -1)
        far = np.where(pb >= 0, by0 - ys_[np.maximum(pb, 0)], h).min()
        return bottom - by1, float(far)

    best = None
    for bh in np.linspace(0.45 * h, 0.03 * h, 300):
        lo, hi = (top + bh / 2, mid) if which == "top" else (mid, bottom - bh / 2)
        if lo >= hi:
            continue
        for _ in range(40):  # solve edge gap == far gap
            yc = (lo + hi) / 2
            e, f = gaps(bh, yc)
            # moving the pair toward the crossing grows the edge gap
            if (e < f) == (which == "top"):
                lo = yc
            else:
                hi = yc
        yc = (lo + hi) / 2
        e, f = gaps(bh, yc)
        if strand_clearance(bh, yc) >= margin and min(e, f) >= margin:
            best = (bh, yc)
            break
    if best is None:
        raise SystemExit(
            f"the {chars!r} rung does not fit in the {which} opening of the X with a "
            f"margin of {margin:.1f}: lower the margin, thin the strands or widen the X"
        )
    bh, yc = best
    if DEBUG:
        e, f = gaps(bh, yc)
        print(f"{chars}: centre x {cx:.1f} (X axis {x0 + w / 2:.1f}); gap to {'cap line' if which == 'top' else 'baseline'} {e:.1f}, "
              f"gap to the strands toward the crossing {f:.1f}; side clearance {strand_clearance(bh, yc):.1f} (margin {margin:.1f}); height {bh / h:.3f} of X")
    s = bh / ink_h
    x = cx - aspect * bh / 2
    parts = []
    for cmd, gx0, gy0, gx1, gy1 in gl:
        yb = yc + (gy1 + gy0) / 2 * s  # glyph ink box centred on yc
        parts.append(f'<path transform="translate({x - gx0 * s:.2f} {yb:.2f}) scale({s:.4f} {-s:.4f})" d="{cmd}" fill="{color}"/>')
        x += (gx1 - gx0) * s + sp * bh
    return "\n".join(parts)


def wordmark(c, uid):
    pad = 40
    track = -10
    x = pad
    top = pad
    base = top + CAP
    parts = []
    for ch in "HELI":
        cmd, adv, _ = glyph(ch)
        parts.append(f'<path transform="translate({x} {base}) scale(1 -1)" d="{cmd}" fill="{c["text"]}"/>')
        x += adv + track
    _, _, (hx0, _, hx1, _) = glyph("H")
    w = hx1 - hx0  # the X is as wide as the H
    stem = 0.21 * CAP
    xs = x + hx0
    parts.append(helix_x(xs, top, w, CAP, stem, c, [("top", "<>"), ("bottom", "{}")], uid, margin=0.03))
    W = xs + w + pad
    H = CAP + 2 * pad
    body = "\n".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="HELIX">\n'
            f"<title>HELIX</title>\n{body}\n</svg>\n")


def icon(c, uid):
    S = 1000
    pad = 90
    h = S - 2 * pad
    w = h * 0.86
    x0 = (S - w) / 2
    stroke = 0.165 * h
    body = helix_x(x0, pad, w, h, stroke, c, [("top", "<>"), ("bottom", "{}")], uid, margin=0.03)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" role="img" aria-label="HELIX">\n'
            f"<title>HELIX</title>\n{body}\n</svg>\n")


args = [a for a in sys.argv[1:] if a != '--debug']
out = Path(args[0]) if args else HERE
for theme, c in PALETTES.items():
    (out / f"helix-wordmark-{theme}.svg").write_text(wordmark(c, f"w{theme}"))
    (out / f"helix-icon-{theme}.svg").write_text(icon(c, f"i{theme}"))
print("written to", out)

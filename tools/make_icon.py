"""Draw the app icon (the same mark as the window header: a split frame with a lit dot) to assets/icon.ico/png."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ACCENT, INK, BG = (200, 245, 96, 255), (232, 234, 240, 255), (12, 13, 16, 255)


def draw(size: int) -> Image.Image:
    s = size / 32
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=7 * s, fill=BG)
    w = max(1, round(2.4 * s))
    d.rounded_rectangle([4 * s, 4 * s, 28 * s, 28 * s], radius=6 * s, outline=INK, width=w)
    d.rectangle([16 * s - w / 2, 4 * s, 16 * s + w / 2, 28 * s], fill=INK)
    d.ellipse([11.3 * s, 11.3 * s, 20.7 * s, 20.7 * s], fill=ACCENT)
    return im


def main() -> None:
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    big = draw(256)
    big.save(out / "icon.png")
    big.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", out / "icon.ico")


if __name__ == "__main__":
    main()

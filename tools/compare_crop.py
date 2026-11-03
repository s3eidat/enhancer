import sys
from PIL import Image
src, out, x, y, w, h, dst = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7]), sys.argv[7]
a = Image.open(src).convert("RGB"); b = Image.open(out).convert("RGB")
s = b.size[0] / a.size[0]
ca = a.crop((x, y, x + w, y + h)).resize((int(w * s), int(h * s)), Image.BICUBIC)
cb = b.crop((int(x * s), int(y * s), int((x + w) * s), int((y + h) * s)))
c = Image.new("RGB", (ca.size[0] * 2 + 8, ca.size[1]), "white"); c.paste(ca, (0, 0)); c.paste(cb, (ca.size[0] + 8, 0)); c.save(dst)

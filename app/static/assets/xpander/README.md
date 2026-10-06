# Xpander page assets

The `/xpander` product page looks for these images here (served from
`/static/assets/xpander/...`):

| File | Where it is used |
|------|------------------|
| `xpander_main_banner.png` | Hero banner (top of the page) — **active** |
| `mitsubishi_xpander_cross_hev_путь_к_приключениям.png` | Alternative hero banner |
| `популярные_темы_mitsubishi_xpander.png` | Visual reference for the topic cards |
| `вибрация_на_d_xpander_у_моря.png` | Current topic card (right column) |
| `автомобильный_чат_mitsubishi_xpander.png` | Chat visual reference |

Drop the PNG files into this directory with exactly these names; the page picks
them up automatically (the markup layers them over a CSS/SVG fallback).

If a file is missing the page falls back to the bundled SVG artwork
(`hero-fallback.svg`, `topic-fallback.svg`) so the layout never looks broken.

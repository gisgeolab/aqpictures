import io
import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import clear_output, display


def ensure_rgb(img):
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)

    if img.ndim == 3 and img.shape[2] >= 3:
        return img[..., :3]

    raise ValueError("Unsupported image shape")


def crop(img, top, left, height, width):
    h, w = img.shape[:2]

    t = max(0, min(top, h - 1))
    l = max(0, min(left, w - 1))

    if height <= 0:
        height = h - t
    if width <= 0:
        width = w - l

    b = max(0, min(t + height, h))
    r = max(0, min(l + width, w))

    return img[t:b, l:r]


def rgb_to_hsv_np(rgb_uint8):
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin + 1e-12

    hue = np.zeros_like(cmax)
    mask = delta > 1e-8

    r_is_max = (cmax == r) & mask
    g_is_max = (cmax == g) & mask
    b_is_max = (cmax == b) & mask

    hue[r_is_max] = ((g - b)[r_is_max] / delta[r_is_max]) % 6.0
    hue[g_is_max] = ((b - r)[g_is_max] / delta[g_is_max]) + 2.0
    hue[b_is_max] = ((r - g)[b_is_max] / delta[b_is_max]) + 4.0
    hue = hue / 6.0

    sat = np.zeros_like(cmax)
    nonzero = cmax > 1e-8
    sat[nonzero] = delta[nonzero] / cmax[nonzero]

    val = cmax

    return np.stack([hue, sat, val], axis=-1)


def hsv_to_rgb_np(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = h * 6.0

    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

    idx = i == 0
    r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]
    idx = i == 1
    r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]
    idx = i == 2
    r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]
    idx = i == 3
    r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]
    idx = i == 4
    r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]
    idx = i == 5
    r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def prepare_view(img_uint8, mode):
    rgb = ensure_rgb(img_uint8)

    if mode == "RGB":
        return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)

    hsv = rgb_to_hsv_np(rgb)

    if mode == "HSV":
        return hsv_to_rgb_np(hsv)

    if mode == "Saturation":
        return hsv[..., 1]

    if mode == "R":
        return rgb[..., 0].astype(np.float32) / 255.0

    if mode == "G":
        return rgb[..., 1].astype(np.float32) / 255.0

    if mode == "B":
        return rgb[..., 2].astype(np.float32) / 255.0

    if mode == "B/R":
        r = rgb[..., 0].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        eps = 1e-6

        ratio = b / (r + eps)

        finite = np.isfinite(ratio)
        lo, hi = np.percentile(ratio[finite], [1, 99]) if finite.any() else (0.0, 1.0)

        if hi <= lo:
            lo, hi = ratio.min(), ratio.max()

        ratio = np.clip(ratio, lo, hi)
        denom = (hi - lo) if hi > lo else 1.0

        return (ratio - lo) / denom

    raise ValueError("Unknown mode")


def show_swipe(ax, imgA, imgB, pos, orientation="Horizontal", alpha_top=1.0):
    ax.clear()
    ax.set_axis_off()

    h = min(imgA.shape[0], imgB.shape[0])
    w = min(imgA.shape[1], imgB.shape[1])

    A = imgA[:h, :w]
    B = imgB[:h, :w]

    if B.ndim == 2:
        ax.imshow(B, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(B)

    if orientation == "Horizontal":
        cut = int(w * pos)
        clip_rect = Rectangle((0, 0), cut, h, transform=ax.transData)
    else:
        cut = int(h * pos)
        clip_rect = Rectangle((0, 0), w, cut, transform=ax.transData)

    if A.ndim == 2:
        imA = ax.imshow(A, cmap="gray", vmin=0, vmax=1, alpha=alpha_top)
    else:
        imA = ax.imshow(A, alpha=alpha_top)

    imA.set_clip_path(clip_rect)


def build_roi_viewer(image_folder):
    image_folder = str(image_folder)
    exts = (".jpg", ".jpeg", ".png")

    compact_slider_layout = widgets.Layout(width="220px")
    compact_dropdown_layout = widgets.Layout(width="260px")

    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(exts)]
    )

    if not image_files:
        raise RuntimeError("No images found in input folder")

    pick1 = widgets.Dropdown(
        options=image_files,
        description="Image 1",
        layout=compact_dropdown_layout
    )

    pick2 = widgets.Dropdown(
        options=image_files,
        description="Image 2",
        layout=compact_dropdown_layout
    )

    top = widgets.IntSlider(
        value=160, min=0, max=2000, step=1,
        description="Top",
        layout=compact_slider_layout
    )

    left = widgets.IntSlider(
        value=116, min=0, max=2000, step=1,
        description="Left",
        layout=compact_slider_layout
    )

    height = widgets.IntSlider(
        value=380, min=0, max=2160, step=1,
        description="Height",
        layout=compact_slider_layout
    )

    width = widgets.IntSlider(
        value=515, min=0, max=3840, step=1,
        description="Width",
        layout=compact_slider_layout
    )

    pos = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.01,
        description="Position",
        layout=compact_slider_layout
    )

    alpha_top = widgets.FloatSlider(
        value=1.0, min=0.2, max=1.0, step=0.05,
        description="Top α",
        layout=compact_slider_layout
    )

    mode_help = {
        "RGB": "RGB: Overall color and brightness.",
        "HSV": "HSV: Visualizes hue, saturation and brightness.",
        "Saturation": "Saturation: Degree of grayness / color intensity (useful for haze).",
        "R": "Red channel intensity.",
        "G": "Green channel intensity.",
        "B": "Blue channel intensity.",
        "B/R": "Blue-to-Red ratio (sensitive to atmospheric scattering)."
    }

    mode = widgets.Dropdown(
        options=["RGB", "HSV", "Saturation", "B/R"],
        value="RGB",
        description="Mode",
        layout=widgets.Layout(width="220px")
    )

    mode_hint = widgets.HTML(
        value=f"<div style='font-size:13px; margin:2px 0 4px 0;'><i>{mode_help['RGB']}</i></div>"
    )

    orientation = widgets.ToggleButtons(
        options=["Horizontal", "Vertical"],
        description="Swipe",

    )

    refresh_btn = widgets.Button(
        description="Refresh",
        layout=widgets.Layout(width="90px", height="32px")
    )

    out = widgets.Image(format="png")

    def update_mode_hint(change):
        mode_hint.value = f"<div style='font-size:13px; margin:2px 0 4px 0;'><i>{mode_help[change['new']]}</i></div>"

    mode.observe(update_mode_hint, names="value")

    def render(_=None):
        f1, f2 = pick1.value, pick2.value

        img1 = iio.imread(os.path.join(image_folder, f1))
        img2 = iio.imread(os.path.join(image_folder, f2))

        img1c = crop(img1, top.value, left.value, height.value, width.value)
        img2c = crop(img2, top.value, left.value, height.value, width.value)

        v1 = prepare_view(img1c, mode.value)
        v2 = prepare_view(img2c, mode.value)

        fig = plt.figure(figsize=(11.5, 9.0))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[1.45, 0.75],
            hspace=0.08,
            wspace=0.03
        )

        ax3 = fig.add_subplot(gs[0, :])
        show_swipe(
            ax3,
            v1,
            v2,
            pos.value,
            orientation.value,
            alpha_top.value
        )

        ax3.set_title(
            f"{f1[:13]} vs {f2[:13]}",
            fontsize=11,
            pad=8
        )

        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        for ax, img, title in [
            (ax1, img1, f1[:13]),
            (ax2, img2, f2[:13]),
        ]:
            ax.imshow(ensure_rgb(img))
            ax.set_axis_off()

            rect = Rectangle(
                (left.value, top.value),
                width.value,
                height.value,
                fill=False,
                edgecolor="red",
                linewidth=2
            )
            ax.add_patch(rect)

        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            top=0.95,
            bottom=0.05,
            hspace=0.10,
            wspace=0.03
        )

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.value = buffer.getvalue()

    for w in [
        pick1, pick2,
        top, left, height, width,
        mode, orientation,
        pos, alpha_top
    ]:
        w.observe(render, names="value")

    refresh_btn.on_click(render)

    controls = widgets.VBox([
        widgets.HBox([pick1, pick2]),
        widgets.HBox([mode, orientation]),
        mode_hint,
        widgets.HBox([pos, alpha_top, top, refresh_btn]),
        widgets.HBox([left, height, width]),
    ])

    clear_output(wait=True)
    display(controls, out)
    render()

    return {
        "top": top,
        "left": left,
        "height": height,
        "width": width,
        "mode": mode
    }
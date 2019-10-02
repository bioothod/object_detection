import numpy as np

from matplotlib import patches, patheffects

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(im)

    #ax.set_xticks(np.linspace(0, 224, 8))
    #ax.set_yticks(np.linspace(0, 224, 8))
    #ax.grid()
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    return fig, ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b, color):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], color=color, fill=False, edgecolor='white', lw=1, linestyle='--'))
    draw_outline(patch, 2)

def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='normal')
    draw_outline(text, 1)

def bb_hw(a):
    # returned coordinates must be x, y, w, h
    #
    # down below is a line for the proceseed and converted format which is x0, y0, x1, y1
    # this is the case for the original COCO dataset (?)
    #return a

    # x0, y0, x1, y1 -> x0, y0, w, h
    return np.array([a[0], a[1], a[2]-a[0]+1, a[3]-a[1]+1])

def draw_im(im, ann, dst, cat_names):
    fig, ax = show_img(im, figsize=(10, 10))

    color_map = {}
    for b, c in ann:
        b = bb_hw(b)

        color = color_map.get(c)
        if color is None:
            color = np.random.rand(3,)
            color_map[c] = color

        draw_rect(ax, b, color)

        if True:
            if c in cat_names:
                draw_text(ax, b[:2], cat_names[c], sz=16)
            else:
                draw_text(ax, b[:2], str(c), sz=16)

    plt.savefig(dst)
    plt.close(fig)

def draw_filename(fn, ann, dst, cat_names):
    im = mpimg.imread(fn)
    draw_im(im, ann, dst, cat_names)

def draw_im_segm(img, masks, centers, dst):
    rows = 1
    columns = 1 + 1 + len(masks)*masks[0].shape[2]

    fig = plt.figure(figsize=(3 * columns, 3 * rows))

    ax = fig.add_subplot(rows, columns, 1)
    ax.set_autoscale_on(True)
    ax.imshow(img)

    ax_img_mask = fig.add_subplot(rows, columns, 2)
    ax_img_mask.set_autoscale_on(True)
    ax_img_mask.imshow(img)

    single_color = False
    ax_idx = 3
    for idx, m in enumerate(masks):
        mask_max = np.max(m, axis=-1)
        for mchannel in range(m.shape[2]):
            ax = fig.add_subplot(rows, columns, ax_idx + idx*m.shape[2] + mchannel)
            ax.set_autoscale_on(True)

            mask = m[:, :, mchannel]
            #mask = np.where(mask > 0, 1, 0).astype(float)

            mask_img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
            if single_color:
                color_mask = np.array([2.0, 166.0, 101.0])/255.
            else:
                color_mask = np.random.random((1, 3)).tolist()[0]

            for ch in range(3):
                mask_img[:, :, ch] = color_mask[ch]

            ax.imshow(np.dstack((mask_img, mask)))
            ax_img_mask.imshow(np.dstack((mask_img, mask * 0.4)))

            if mchannel == 4:
                for c in centers:
                    #cr = plt.Circle((c[1], c[0]), 2, color='r')
                    cr = plt.Circle(c, 2, color='r')
                    ax.add_artist(cr)

                    cr = plt.Circle(c, 2, color='r')
                    #cr = plt.Circle((c[1], c[0]), 2, color='r')
                    ax_img_mask.add_artist(cr)

    plt.savefig(dst)
    plt.close(fig)

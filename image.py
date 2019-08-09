import numpy as np

from matplotlib import patches, patheffects

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(im)

    ax.set_xticks(np.linspace(0, 224, 8))
    ax.set_yticks(np.linspace(0, 224, 8))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    return fig, ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=1))
    draw_outline(patch, 2)

def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='normal')
    draw_outline(text, 1)

def bb_hw(a):
    #return a
    #return np.array([a[1], a[0], a[3]-a[1]+1, a[2]-a[0]+1])
    return np.array([a[0], a[1], a[2]-a[0]+1, a[3]-a[1]+1])

def draw_im(im, ann, dst, cat_names):
    fig, ax = show_img(im, figsize=(10, 10))
    for b, c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cat_names[c], sz=16)

    plt.savefig(dst)
    plt.close(fig)

def draw_filename(fn, ann, dst, cat_names):
    im = mpimg.imread(fn)
    draw_im(im, ann, dst, cat_names)

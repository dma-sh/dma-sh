import numpy as np
import os
import warnings
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import ticker


COLORS = (
    '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
    '#001177', '#117700', '#990022', '#885500', '#553366', '#006666'
)

COLORS_GRAY = (
    '#000000', '#71797E', '#D3D3D3'
    # '#000000', '#B2BEB5', '#D3D3D3'
)

LINESTYLES = (
    '-', '--', ':', '-.', (5, (10, 3)), (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 1, 1, 1, 1, 1)), (0, (5, 10)), (0, (5, 1)), (0, (3, 10, 1, 10, 1, 10))
)


def save(fig, filename):
    filename = pathlib.Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    # png = filename.with_suffix('.png')
    pdf = filename.with_suffix('.pdf')
    # fig.savefig(png, dpi=300, bbox_inches="tight")
    # print('Saved', png.name)
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    print('Saved', pdf.name)
    # If pdfcrop from texlive is not available, the below prints a warning.
    os.system(f'pdfcrop {pdf} {pdf}')


def plots(
        amount, cols=4, size=(2, 2.3), xticks=2, yticks=2, grid=(2, 2), **kwargs):
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    # BIGGER_SIZE = 12

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": SMALL_SIZE,
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": MEDIUM_SIZE,
    })
    cols = min(amount, cols)
    rows = int(np.ceil(amount / cols))
    size = (cols * size[0], rows * size[1])
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, **kwargs)
    axes = axes.flatten()
    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(xticks))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(yticks))
        if grid:
            grid = (grid, grid) if not hasattr(grid, '__len__') else grid
            ax.grid(which='both', color='#eeeeee')
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(int(grid[0])))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(int(grid[1])))
            ax.tick_params(which='minor', length=0)
    for ax in axes[amount:]:
        ax.axis('off')
    return fig, axes


def curve(
        ax, domain, values, low=None, high=None, label=None, order=0, **kwargs):
    finite = np.isfinite(values)
    ax.plot(
        domain[finite], values[finite],
        label=label, zorder=1000 - order, **kwargs)
    if low is not None:
        # pop linestyle, as dashes will lead to errors with pdf savefig and lw=0
        kwargs.pop('ls', None)
        kwargs.pop('linestyle', None)
        kwargs.pop('linestyles', None)
        kwargs.pop('dashes', None)
        ax.fill_between(
            domain[finite], low[finite], high[finite],
            zorder=100 - order, alpha=0.2, lw=0, **kwargs)


def bars(
        ax, labels, values, lower=None, upper=None, colors=None, reverse=False):
    values = np.array(values)
    domain = np.arange(len(values))
    assert values.shape == domain.shape, (values.shape, domain.shape)
    assert len(labels) == len(values), (labels, values)
    if reverse:
        labels = labels[::-1]
        values = values[::-1]
        lower = lower[::-1]
        upper = upper[::-1]
        colors = colors[::-1]
    yerr = np.stack([-(lower - values), upper - values], 0)
    ax.bar(domain, values, yerr=yerr, color=colors or COLORS[len(labels)])
    ax.set_xticks(domain + 0.3)
    ax.set_xticklabels(labels, ha='right', rotation=30, rotation_mode='anchor')
    ax.set_xlim(-0.6, domain[-1] + 0.4)
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='major', length=2, labelsize=9, pad=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def legend(fig, data=None, adjust=True, plotpad=0.5, legendpad=0, **kwargs):
    options = dict(
        fontsize='medium', numpoints=1, labelspacing=0, columnspacing=1.2,
        handlelength=2.2, handletextpad=0.5, ncol=4, loc='lower center')
    options.update(kwargs)
    # Find all labels and remove duplicates.
    entries = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if data and label in data:
                label = data[label]
            entries[label] = handle
    leg = fig.legend(list(reversed(entries.values())), list(reversed(entries.keys())), **options)
    leg.get_frame().set_edgecolor('white')
    leg.set_bbox_to_anchor((0.5, -0.04))
    if adjust:
        legextent = leg.get_window_extent(fig.canvas.get_renderer())
        legextent = legextent.transformed(fig.transFigure.inverted())
        yloc, xloc = options['loc'].split()
        legpad = legendpad
        xpad, ypad = legpad if hasattr(legpad, '__len__') else (legpad,) * 2
        x0 = dict(left=legextent.x1 + xpad, center=0, right=0)[xloc]   # left
        y0 = dict(lower=legextent.y1 + ypad, center=0, upper=0)[yloc]  # bottom
        x1 = dict(left=1, center=1, right=legextent.x0 - xpad)[xloc]   # right
        y1 = dict(lower=1, center=1, upper=legextent.y0 - ypad)[yloc]  # top
        rect = [x0, y0, x1, y1]  # left, bottom, right, top
        xpad, ypad = plotpad if hasattr(plotpad, '__len__') else (plotpad,) * 2
        fig.tight_layout(rect=rect, w_pad=xpad, h_pad=ypad)


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
    assert fill in ('nan', 'last', 'zeros')
    xs = xs if isinstance(xs, np.ndarray) else np.asarray(xs)
    ys = ys if isinstance(ys, np.ndarray) else np.asarray(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    binned = []
    for start, stop in zip(borders[:-1], borders[1:]):
        left = (xs < start).sum()
        right = (xs <= stop).sum()
        value = np.nan
        if left < right:
            value = reduce(ys[left:right], reducer)
        if np.isnan(value):
            if fill == 'zeros':
                value = 0
            if fill == 'last' and binned:
                value = binned[-1]
        binned.append(value)
    binned.append(ys[-1])
    return borders, np.array(binned)


def tensor(runs, borders, tasks=None, methods=None, seeds=None, fill='nan'):
    tasks = tasks or sorted(set(run['task'] for run in runs))
    methods = methods or sorted(set(run['method'] for run in runs))
    seeds = seeds or sorted(set(run['seed'] for run in runs))
    # tensor = np.empty((len(tasks), len(methods), len(seeds), len(borders) - 1))
    tensor = np.empty((len(tasks), len(methods), len(seeds), len(borders)))
    tensor[:] = np.nan
    for run in runs:
        try:
            i = tasks.index(run['task'])
            j = methods.index(run['method'])
            k = seeds.index(run['seed'])
        except ValueError:
            continue
        _, ys = binning(run['xs'], run['ys'], borders, fill=fill)
        tensor[i, j, k, :] = ys
    return tensor, tasks, methods, seeds


def reduce(values, reducer=np.nanmean, *args, **kwargs):
    with warnings.catch_warnings():  # Buckets can be empty.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return reducer(values, *args, **kwargs)


def smart_format(x, pos=None):
    if abs(x) < 1e3:
        if float(int(x)) == float(x):
            return str(int(x))
        return str(round(x, 10)).rstrip('0')
    if abs(x) < 1e6:
        return f'{x/1e3:.0f}K' if x == x // 1e3 * 1e3 else f'{x/1e3:.1f}K'
    if abs(x) < 1e9:
        return f'{x/1e6:.0f}M' if x == x // 1e6 * 1e6 else f'{x/1e6:.1f}M'
    return f'{x/1e9:.0f}B' if x == x // 1e9 * 1e9 else f'{x/1e9:.1f}B'


def representation_fig(zs, labels, title=''):
    uniq_labels = np.unique(labels)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for i, lab in enumerate(uniq_labels):
        idxs = labels.squeeze() == lab
        color_i = i * len(COLORS) // len(uniq_labels)
        ax.scatter(zs[idxs, 0], zs[idxs, 1], label=lab, color=COLORS[color_i])

        xtext, ytext = np.median(zs[idxs, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend()

    return fig

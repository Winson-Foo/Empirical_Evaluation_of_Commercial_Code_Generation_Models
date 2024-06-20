def plot_image(img, cmap='gray', normalize=False, vmin=None, vmax=None, ax=None):
    if normalize:
        vmin, vmax = img.min(), img.max()
    elif vmin is None or vmax is None:
        vmin, vmax = 0, 255

    if ax is None:
        ax = plt.gca()
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis('off')


def plot_lines(cuts, img_shape, ax=None):
    if ax is None:
        ax = plt.gca()
    for c in cuts:
        if c.orientation == 'v':
            lines = ax.plot([c.idx, c.idx], [0, img_shape[0] - 0.5])
        elif c.orientation == 'h':
            lines = ax.plot([0, img_shape[1] - 0.5], [c.idx, c.idx])
        plt.setp(lines, color=c.color)
        plt.setp(lines, linewidth=1)
    ax.axis('off')


def plot_curve(cuts, img, cmap='gray', normalize=False, vmin=None, vmax=None, ax=None):
    if normalize:
        vmin, vmax = img.min(), img.max()
    elif vmin is None or vmax is None:
        vmin, vmax = 0, 255

    if ax is None:
        ax = plt.gca()

    def test(x):
        return (x * 0.5).sum()

    for c in cuts:
        if c.orientation == 'v':
            curve = img[:, c.idx]
        elif c.orientation == 'h':
            curve = img[c.idx, :]
        ax.plot(curve, color=c.color, alpha=0.5)


def plot_patches(img, patches, selection=[], cmap='gray', normalize=False, vmin=None, vmax=None):
    if normalize:
        vmin, vmax = img.min(), img.max()
    elif vmin is None or vmax is None:
        vmin, vmax = 0, 255

    fig, ax = plt.subplots()
    plot_image(img, cmap=cmap, normalize=normalize, vmin=vmin, vmax=vmax, ax=ax)
    for p in patches:
        if p.color == 'w':
            zorder = 1
        else:
            zorder = 2
        rect = mpatches.Rectangle((p.box[1], p.box[0]),
                                  p.box[2],
                                  p.box[3],
                                  linewidth=2,
                                  edgecolor=p.color,
                                  facecolor='none',
                                  zorder=zorder)
        ax.add_artist(rect)

    if len(selection) > 0:
        nrows = int(np.ceil(np.sqrt(len(selection))))
        ncols = int(np.ceil(np.sqrt(len(selection))))
        fig, grid = plt.subplots(nrows, ncols)

        for ax, idx in zip(grid, selection):
            p = patches[idx]
            crop = img[p.box[0]:p.box[0] + p.box[2], p.box[1]:p.box[1] + p.box[3]]
            plot_image(crop, cmap=cmap, normalize=normalize, vmin=vmin, vmax=vmax, ax=ax)

        plt.tight_layout()

class ImagePlotter:
    def __init__(self, img, cmap='gray', normalize=False, vmin=None, vmax=None):
        self.img = img
        self.cmap = cmap
        self.normalize = normalize
        self.vmin = vmin
        self.vmax = vmax

    def plot_image(self, ax=None):
        if self.normalize:
            vmin, vmax = self.img.min(), self.img.max()
        elif self.vmin is None or self.vmax is None:
            vmin, vmax = 0, 255
        else:
            vmin, vmax = self.vmin, self.vmax

        if ax is None:
            ax = plt.gca()
        ax.imshow(self.img, vmin=vmin, vmax=vmax, cmap=self.cmap)
        ax.axis('off')

    def plot_lines(self, cuts, ax=None):
        if ax is None:
            ax = plt.gca()
        for c in cuts:
            if c.orientation == 'v':
                lines = ax.plot([c.idx, c.idx], [0, self.img.shape[0] - 0.5])
            elif c.orientation == 'h':
                lines = ax.plot([0, self.img.shape[1] - 0.5], [c.idx, c.idx])
            plt.setp(lines, color=c.color)
            plt.setp(lines, linewidth=1)
        ax.axis('off')

    def plot_curve(self, cuts, ax=None):
        if ax is None:
            ax = plt.gca()

        for c in cuts:
            if c.orientation == 'v':
                curve = self.img[:, c.idx]
            elif c.orientation == 'h':
                curve = self.img[c.idx, :]
            ax.plot(curve, color=c.color, alpha=0.5)

    def plot_patches(self, patches, selection=[], ax=None):
        if ax is None:
            ax = plt.gca()

        self.plot_image(ax=ax)
        for p in patches:
            if p.color == 'w':
                zorder = 1
            else:
                zorder = 2
            rect = mpatches.Rectangle((p.box[1], p.box[0]),
                                      p.box[2],
                                      p.box[3],
                                      linewidth=2,
                                      edgecolor=p.color,
                                      facecolor='none',
                                      zorder=zorder)
            ax.add_artist(rect)

        if len(selection) > 0:
            nrows = int(np.ceil(np.sqrt(len(selection))))
            ncols = int(np.ceil(np.sqrt(len(selection))))
            fig, grid = plt.subplots(nrows, ncols)

            for ax, idx in zip(grid, selection):
                p = patches[idx]
                crop = self.img[p.box[0]:p.box[0] + p.box[2], p.box[1]:p.box[1] + p.box[3]]
                ImagePlotter(crop, cmap=self.cmap, normalize=self.normalize, vmin=self.vmin, vmax=self.vmax).plot_image(ax=ax)

            plt.tight_layout()
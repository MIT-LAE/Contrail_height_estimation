import os
import math
import datetime as dt
from typing import List

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patches as patches

# Default time locator and formatter settings for class `TimeLocator`
TIMELOCATOR_STEPS = [1, 2, 5, 10, 15, 30, 300, 600, 900]
TIMELOCATOR_MINORSTEPS = [0.2, 0.5, 1, 2, 3, 5, 10, 60, 120, 300]

class TimeLocator(mpl.ticker.Locator):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    def __init__(self, n: int, time: List[dt.datetime],
                steps: List[int]=TIMELOCATOR_STEPS,
                 minorsteps: List[float]=TIMELOCATOR_MINORSTEPS):
        
        self.n = int(n)
        self.time = time
        self.steps = np.array(steps, np.float64)
        self.minorsteps = np.array(minorsteps, np.float64)
        self.minorlocs = [] 
        
    def __call__(self):
        
        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = max(int(vmin), 0), min(int(vmax), len(self.time)-1)
        
        # Time difference
        time_diff = self.time[vmax] - self.time[vmin]
        second_diff = time_diff.days * 3600 * 24 + time_diff.seconds \
                        + time_diff.microseconds * 10e-6
        if second_diff < 0:
            second_diff *= -1
        
        ratio = 1.0 *(vmax - vmin) / second_diff
        
        time0 = self.time[vmin]
        
        offset = - ratio * (time0.microsecond * 10e-6\
                            + time0.second\
                            + time0.minute * 60 - 0.2)
        
        stepdiffs = self.steps - second_diff/self.n
        np.place(stepdiffs, stepdiffs<0, np.inf)
        i = stepdiffs.argmin()
        base = ratio * self.steps[i]
        minorbase = ratio * self.minorsteps[i]
        
        offset = offset % base
        self.minorlocs = np.arange(vmin + offset - base, vmax + offset + base,
                            minorbase)
        return np.arange(vmin + offset, vmax + offset, base)
        
        
class TimeMinorLocator(mpl.ticker.Locator):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    def __call__(self):
        locator = self.axis.get_major_locator()
        if isinstance(locator, TimeLocator):
            return locator.minorlocs
        else:
            return []


class TimeFormatter(mpl.ticker.Formatter):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    def __init__(self, time):
        self.time = time

    def __call__(self, x, pos=None):
        i = int(x)
        if i >= len(self.time) or i < 0: return "undef"
        return self.time[i].strftime("%H:%M:%S")


def new_axes(fig, x, y, width, height, padding=1.0):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """

    figw_old, figh_old = fig.get_size_inches()

    figw = max(figw_old, x + width + padding)
    figh = max(figh_old, y + height + padding)

    resize_figure(fig, figw, figh)
    return fig.add_axes([x/figw, 1-(y+height)/figh, width/figw, height/figh])


def resize_figure(fig, figw, figh):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    figw_old, figh_old = fig.get_size_inches()

    fig.set_size_inches(figw, figh)

    xratio = figw_old/figw
    yratio = figh_old/figh

    for ax in fig.axes:
        xrel, yrel, wrel, hrel = ax.get_position(True).bounds
        yrel = 1-yrel
        ax.set_position([xrel*xratio, 1-yrel*yratio, wrel*xratio, hrel*yratio])

    
def get_axes_bounds(fig, axes):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    figw, figh = fig.get_size_inches()
    xrel, yrel, wrel, hrel = axes.get_position(True).bounds
    return figw*xrel, figh*(1-yrel-hrel), figw*wrel, figh*hrel
    

def fit_colorbar(fig, axes, aspect=0.03, space=0.4, padding=0.0):
    """
   
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot

    Creates new axes for a colorbar at the expense of main axes.
    Arguments:
        fig     -- an instance of mpl.Figure
        axes    -- an instance of mpl.Axes
        aspect  -- colorbar axes aspect ratio
    Returns:
        An instance of mpl.Axes.
    """
    x, y, width, height = get_axes_bounds(fig, axes)
    return new_axes(fig, x + width + space, y, aspect*height, height,
                    padding=padding)
        
def loadcolormap(filename, name):
    """"
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot

    Returns a tuple of matplotlib colormap, matplotlib norm,
    and a list of ticks loaded from the file filename in format:
    BOUNDS
    from1 to1 step1
    from2 to2 step2
    ...
    TICKS
    from1 to1 step1
    from2 to2 step2
    COLORS
    r1 g1 b1
    r2 g2 b2
    ...
    UNDER_OVER_BAD_COLORS
    ro go bo
    ru gu bu
    rb gb bb
    Where fromn, ton, stepn are floating point numbers as would be supplied
    to numpy.arange, and rn, gn, bn are the color components the n-th color
    stripe. Components are expected to be in base10 format (0-255).
    UNDER_OVER_BAD_COLORS section specifies colors to be used for
    over, under and bad (masked) values in that order.
    Arguments:
        filename    -- name of the colormap file
        name        -- name for the matplotlib colormap object
    Returns:
        A tuple of: instance of ListedColormap, instance of BoundaryNorm, ticks
    """
    CCPLOT_CMAP_PATH = ""

    bounds = []
    ticks = []
    rgbarray = []
    specials = []
    mode = "COLORS"

    fp = None
    if filename.startswith("/") or \
       filename.startswith("./") or \
       filename.startswith("../"):
        try:
            fp = open(filename, "r")
        except IOError as err:
            raise FileNotFoundError
    else:
        for path in CCPLOT_CMAP_PATH.split(":"):
            try:
                fp = open(os.path.join(path, filename), "r")
            except IOError as err: continue
            break

    try:
        lines = fp.readlines()
        for n, s in enumerate(lines):
            s = s.strip()
            if len(s) == 0: continue
            if s in ("BOUNDS", "TICKS", "COLORS", "UNDER_OVER_BAD_COLORS"):
                mode = s
                continue

            a = s.split()
            if len(a) not in (3, 4):
                raise ValueError("Invalid number of fields")

            if mode == "BOUNDS":
                bounds += list(np.arange(float(a[0]), float(a[1]),
                                float(a[2])))
            elif mode == "TICKS":
                ticks += list(np.arange(float(a[0]), float(a[1]), float(a[2])))
            elif mode == "COLORS":
                rgba = [int(c)/256.0 for c in a]
                if len(rgba) == 3: rgba.append(1)
                rgbarray.append(rgba)
            elif mode == "UNDER_OVER_BAD_COLORS":
                rgba = [int(c)/256.0 for c in a]
                if len(rgba) == 3: rgba.append(1)
                specials.append(rgba)

    except IOError as err:
        fail(err)
    except ValueError as err:
        fail("Error reading `%s' on line %d: %s" % (filename, n+1, err))

    if (len(rgbarray) > 0):
        colormap = mpl.colors.ListedColormap(rgbarray, name)
        try:
            colormap.set_under(specials[0][:3], specials[0][3])
            colormap.set_over(specials[1][:3], specials[1][3])
            colormap.set_bad(specials[2][:3], specials[2][3])
        except IndexError: pass
    else:
        colormap = None

    if len(bounds) == 0:
        norm = None
    else:
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
    if len(ticks) == 0: ticks = None
    return (colormap, norm, ticks)


def lon2str(lonf, degree=""):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    if lonf >= 0.0: return "%.2f%sE" % (lonf, degree)
    else: return "%.2f%sW" % (-lonf, degree)


def lat2str(latf, degree=""):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    if latf >= 0.0: return "%.2f%sN" % (latf, degree)
    else: return "%.2f%sS" % (-latf, degree)


def setup_lonlat_axes(fig, axes, lon, lat, axis="x"):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    @mpl.ticker.FuncFormatter
    def lonlat_formatter(x, pos=None):
        i = int(x)
        if x < 0 or x >= len(lon): return ""
        return "%s\n%s" % (lon2str(lon[i], "$\degree$"), \
                           lat2str(lat[i], "$\degree$"))

    if axis == 'x':
        llaxes = axes.twiny()
        llaxes.set_xlim(axes.get_xlim())
        llaxes.xaxis.set_major_locator(CopyLocator(axes.xaxis))

        for tick in llaxes.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)
            tick.tick2line.set_visible(True)
            tick.label2.set_visible(True)

        for line in llaxes.xaxis.get_ticklines():
            line.set_marker(mpl.lines.TICKUP)

        for label in llaxes.xaxis.get_ticklabels():
            label.set_y(label.get_position()[1] + 0.005)

        llaxes.xaxis.set_major_formatter(lonlat_formatter)
    else:
        llaxes = axes.twinx()
        llaxes.set_ylim(axes.get_ylim())
        llaxes.yaxis.set_major_locator(CopyLocator(axes.yaxis))

        for tick in llaxes.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)
            tick.tick2line.set_visible(True)
            tick.label2.set_visible(True)

        for line in llaxes.yaxis.get_ticklines():
            line.set_marker(mpl.lines.TICKRIGHT)

        llaxes.yaxis.set_major_formatter(lonlat_formatter)


class CopyLocator(mpl.ticker.Locator):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    def __init__(self, axis):
        self.model_axis = axis

    def __call__(self):
        return self.model_axis.get_majorticklocs()

    
class SciFormatter(mpl.ticker.Formatter):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    def __call__(self, x, pos=None):
        if x == 0.0: return "0.0"
        y = math.log(abs(x), 10)
        n = int(math.floor(y))
        if n < -1 or n > 2: return "%.1fx10$^{%d}$" % (x/10**n, n)
        else: return "%.1f" % (x,)


def plot_caliop_curtain(fig, ax, lons, lats, times, data, min_alt=0.0,
                                max_alt=40.0,
                                colorbar=False, dist_axis=False, rotate=False,
                                add_scalebar=False, **kwargs):
    """
    Plot a CALIOP curtain, i.e. a 2D array of data with time and altitude
    as axes. The data is plotted as a color map.

    Parameters
    ----------
    fig : mpl.Figure
        Figure instance
    ax : mpl.Axes
        Axes instance
    lons : np.array
        Array of longitudes, length equal to number of columns in `data`
    lats : np.array
        Array of latitudes, length equal to number of columns in `data`
    times : np.array
        Array of times, length equal to number of columns in `data`
    data : np.array
        2D array of data, rows correspond to altitude dimension
    min_alt : float
        Minimum altitude to plot, in km
    max_alt : float
        Maximum altitude to plot, in km
    colorbar : bool
        Whether to plot a colorbar
    dist_axis : bool
        Whether to plot an additional horizontal axis with the distance
        traversed since the first profile within the curtain
    rotate : bool
        Whether to rotate the plot 90 degrees. Default is False, 
        corresponding to the altitude axis being vertical.
    add_scalebar : bool
        Whether to add a scalebar to the plot, which can be used to estimate
        the distance between profiles.
    """
    
    # Load standard CALIPSO backscatter colormap
    if kwargs.get("cmap", None) is None: 
        cmap_path = os.path.join(os.path.dirname(__file__),
                    "assets/calipso-backscatter.cmap")
        cmap, norm, ticks = loadcolormap(cmap_path, "CALIOP")
    # Use user-specified colormap
    else:
        cmap = kwargs.get("cmap")
        norm = kwargs.get("norm")
        ticks = kwargs.get("ticks", None)

    ve1 = min_alt
    ve2 = max_alt

    if rotate:
        data = data.T[::-1,:]
        im_extent = (ve2, ve1, 0, data.shape[1])
    else
        im_extent = (0, data.shape[1], ve1, ve2)

    im = ax.imshow(data, cmap=cmap, norm=norm, extent=im_extent,
                    interpolation='nearest')
    ax.set_aspect('auto')

    # Get fig and ax dimensions
    x, y, width, height = get_axes_bounds(fig, ax) 
    figw, figh = fig.get_size_inches()

    # Set up time axis
    diff = (times[-1] - times[0])
    nseconds = diff.days * 24 * 3600 + diff.seconds
    
    if rotate:
        ax.set(ylabel="Time (UTC)")
        time_axis = ax.yaxis
    else:
        ax.set(xlabel="Time (UTC)")
        time_axis = ax.xaxis
    
    time_axis.set_minor_locator(TimeMinorLocator())
    time_axis.set_major_locator(TimeLocator(5, times))
    time_axis.set_major_formatter(TimeFormatter(times))

    # Rotate axis labels if plot is rotated
    if rotate:
        for label in ax.yaxis.get_ticklabels():
            label.set_x(-0.05/figw)
            label.set_rotation("horizontal")
        time_axis.get_label().set_rotation('vertical')
    
    else:
        for line in ax.xaxis.get_ticklines() + ax.xaxis.get_minorticklines():
            line.set_marker(mpl.lines.TICKDOWN)
        for label in ax.xaxis.get_ticklabels():
            label.set_y(-0.05/figh)
    
    # Configure altitude axis
    if rotate:
        ax.set_xlabel("Altitude (km)")
        alt_axis = ax.xaxis
    else:
        ax.set_ylabel("Altitude (km)")
        alt_axis = ax.yaxis
    
    majorticksbases = np.array([0.5, 1, 2, 5])
    minorticksbases = np.array([0.1, 0.2, 0.5, 1])

    # TODO : what are these numbers?
    height_per_tick = (ve2-ve1)/(height/(12*2/72.0))
    
    i = np.argmin(np.abs(majorticksbases - height_per_tick))
    minor_locator = mpl.ticker.MultipleLocator(minorticksbases[i])
    major_locator = mpl.ticker.MultipleLocator(majorticksbases[i])
    alt_axis.set_minor_locator(minor_locator)
    alt_axis.set_major_locator(major_locator)

    for label in alt_axis.get_ticklabels():
        if rotate:
            label.set_y(-0.05/figh)
        else:
            label.set_x(-0.05/figw)

    for line in alt_axis.get_ticklines() + alt_axis.get_minorticklines():
        if rotate:
            line.set_marker(mpl.lines.TICKDOWN)
        else:
            line.set_marker(mpl.lines.TICKRIGHT)
    
    # Hide ticks on the top and right-hand side.

    for tick in ax.xaxis.get_major_ticks() + \
                ax.yaxis.get_major_ticks() + \
                ax.xaxis.get_minor_ticks() + \
                ax.yaxis.get_minor_ticks():
        tick.tick1line.set_visible(True)
        tick.label1.set_visible(True)
        tick.tick2line.set_visible(False)
        tick.label2.set_visible(False)
            

    if colorbar:
        cbaxes = fit_colorbar(fig, ax, space=0.4, padding=1.0)
        orientation = "horizontal" if rotate else "vertical"
        cb = fig.colorbar(im, ax=ax, cax=cbaxes, orientation=orientation,
                        extend="both", ticks=ticks)

        cb.ax.tick_params(direction="in")
        for label in cb.ax.get_yticklabels():
            label.set_fontsize(8)


    if dist_axis:
        if not rotate:
            setup_dist_axes(fig, ax, times)
        else:
            raise NotImplementedError(("Distance axis not implemented for "
                                    "rotated plots"))
    
    
    setup_lonlat_axes(fig, ax, lons, lats, axis='y' if rotate else 'x')

    # TODO: what are these numbers?
    if add_scalebar:
        if not rotate:
            ax.add_patch(patches.Rectangle([14.0, 0], height=100+2*33.3,
                                            width=1.0, facecolor="w"))
            ax.plot([14.3, 14.3], [50, 50+2*33.3], c="k")
            ax.text(14.3, 50+33.3, "20 km", ha="right", va="center",
                    rotation=90, c="k", fontsize=6)
        
        else:
            ax.add_patch(patches.Rectangle([0, 14.5], width=100+2*33.3,
                                            height=0.5, facecolor="w"))
            ax.plot([50, 50+2*33.3], [14.7, 14.7], c="k")
            ax.text(50+33.3, 14.7, "20 km", ha="center", va="bottom",
                        rotation=0, c="k", fontsize=6)
        
    
    

def setup_dist_axes(fig, axes, times, axis="x"):
    """
    Source: https://github.com/peterkuma/ccplot/blob/master/bin/ccplot
    """
    
    dists = 7*np.cumsum((times[1:].astype(np.datetime64)\
                     -times[:-1].astype(np.datetime64))/np.timedelta64(1,'s'))
    
    
    @mpl.ticker.FuncFormatter
    def dist_formatter(x, pos=None):
        i = int(x)
        if x < 0 or x >= len(dists): return ""
        return f"{int(dists[i])}"

    if axis == 'x':
        llaxes = axes.twiny()
        llaxes.set_xlim(axes.get_xlim())
        llaxes.xaxis.set_major_locator(CopyLocator(axes.xaxis))

        # Move twinned axis ticks and label from top to bottom
        llaxes.xaxis.set_ticks_position("bottom")
        llaxes.xaxis.set_label_position("bottom")
        llaxes.spines.bottom.set_position(('axes', -0.15))

        llaxes.set(xlabel="Distance, km")
        llaxes.xaxis.set_major_formatter(dist_formatter)
    else:
        llaxes = axes.twinx()
        llaxes.set_ylim(axes.get_ylim())
        llaxes.yaxis.set_major_locator(CopyLocator(axes.yaxis))

        for tick in llaxes.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)
            tick.tick2line.set_visible(True)
            tick.label2.set_visible(True)

        for line in llaxes.yaxis.get_ticklines():
            line.set_marker(mpl.lines.TICKRIGHT)


        llaxes.yaxis.set_major_formatter(dist_formatter)


def add_scalebar(ax, length, label):
    scalebar = AnchoredSizeBar(ax.transData, length, label, 'upper left',
                                pad=0.5, frameon=False, color="w")
    ax.add_artist(scalebar)

def rotate_all_labels(axes):
    
    for ax in axes:
        
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation("vertical")
        for label in ax.yaxis.get_ticklabels():
            label.set_rotation("vertical")
            
        ax.xaxis.get_label().set_rotation('vertical')
        ax.yaxis.get_label().set_rotation('vertical')
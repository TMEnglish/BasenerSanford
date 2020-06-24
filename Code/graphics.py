# %load ./Code/graphics.py
from IPython.display import Image, display, HTML

# Use the Seaborn package to generate plots.
import seaborn as sns
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
sns.set_style("darkgrid", {"axes.facecolor": ".92"})
sns.set_palette(sns.color_palette("Set2", 8))


def save_and_display(figure, filename, format='png', dpi=600, close=True):
    """
    Displays figure after saving it with the given attributes.
    
    If `close` is true, then the plot of the figure is closed.
    """
    figure.savefig(filename, format=format, dpi=dpi)
    display(Image(filename=filename))
    if close:
        plt.close(figure)


def exp_latex(value, prefix=''):
    """
    Returns LaTeX expression of `value` as integer power of 2 or 10.
    """
    value = float(value)
    base = 2
    log = round(mp.log(value, base))
    if value != base**log:
        base = 10
        log = round(mp.log(value, base))
        if value != base**log:
            message = '`value` is not an integer power of 2 or 10'
            raise ValueError(message)
    string = '${}{}^{}{}{}$'.format(prefix, base, '{', log, '}')
    return string


def bs_plot(solutions, m, intermediate, ls=['-', '-', '-'], ax=None):
    """
    Plots normalized `solutions` indexed 0, `intermediate`, and -1.
    
    Returns the figure and the axis system for the plot.
    
    It is assumed that `solutions` is a 2-dimensional array of non-
    negative numbers, that the sum of elements for each row is positive,
    and that `solutions[n]` corresponds to year `n`.
    
    The linestyles for the three plotted solutions are given by `ls`.
    If an axis system `ax` is supplied, then the plot is on that axis
    system. Otherwise, a new figure and axis system are created.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Relative Frequency')
    years = [0, intermediate, len(solutions) - 1]
    for year, style in zip(years, ls):
        s = solutions[year] / fsum(solutions[year])
        label = 'Year {}'.format(year)
        ax.plot(m[s>0], s[s>0], label=label, ls=style)
    ax.legend(loc='best', fontsize='small')
    return ax.get_figure(), ax
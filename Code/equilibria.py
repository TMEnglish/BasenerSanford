import scipy.stats as stats

class Equilibria(object):
    """
    Container for an array of equilibria and associated statistics.
    
    Each row of the array corresponds to a distribution of probability
    over mutational effects, and each column corresponds to an upper
    limit on fitness.
    """    
    def __init__(self, mixture_weights, b_maxes, params_class=None):
        """
        Create array of equilibria, and calculate associated statistics.
        
        `params_class`: subclass of `Parameters` (default `Parameters`)
        `mixture_weights`: weightings of beneficial effects
        `b_maxes`: upper limits on birth rate
        
        The mixture weights must be integer powers of 2 or 10.
        """
        # Create array of parameters objects, with one row for each
        # mixture weight, and one column for each birth-rate limit.
        if params_class is None:
            params_class = Parameters
        params = [[params_class(b_max=b_max, gamma=gamma) 
                       for b_max in b_maxes]
                           for gamma in mixture_weights]
        self.params = np.array(params)
        #
        # Get the upper limits on fitness from the first row of the
        # parameters objects.
        fitness_limits = [p.m[-1] for p in params[0]]
        self.fitness_limits = np.array(fitness_limits, dtype=float)
        #
        # Convert the mixture weights to LaTeX expressions that will
        # be used in labeling figures.
        self.mixture_weights = mixture_weights.astype(float)
        self.weight_labels = [exp_latex(g, '\gamma=') 
                                  for g in mixture_weights]
        #
        # Create arrays to hold results, with rows corresponding to
        # probability distributions over mutational effects, and columns
        # corresponding to upper limits on fitness.
        m, n = self.params.shape
        self.eq = np.empty((m, n), dtype=object)
        self.e_value = np.empty((m, n))
        self.eigen_error = np.empty((m, n))
        self.mean = np.empty((m, n))
        self.var = np.empty((m, n))
        #
        # Calculate equilibria along with means/variances of fitnesses.
        for i in range(m):
            for j in range(n):
                W = self.params[i,j].W.astype(float)
                m = self.params[i,j].m
                e_value, e_vector, error = equilibrium(W)
                e_vector /= fsum(e_vector)
                self.eq[i,j] = e_vector
                self.e_value[i,j] = e_value
                self.eigen_error[i,j] = error
                self.mean[i,j], self.var[i,j] = mean_var(self.eq[i,j], m)

    def plot_stats(self):
        """
        Plot the means and variances in fitness for all equilibria.
        """
        # Set up two axis systems, one for means, the other for variances.
        sns.set_palette(sns.color_palette("Set2", len(self.mean)))
        fig, ax_m = plt.subplots()
        ax_v = ax_m.twinx()
        ax_v.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax_v.grid(False)
        #
        # Plot the means and variances as functions of the upper
        # limit on fitness, row by row. Each row corresponds to a
        # distribution of probability over mutational effects.
        lims = self.fitness_limits
        for m, v, l in zip(self.mean, self.var, self.weight_labels):
            ax_m.plot(lims, m, label=l, marker='+', ls='none')
            ax_v.plot(lims, v, label=l, marker='o', mfc='none', ls='none')
        #
        # Fit lines to means by linear regression, and plot them.
        xlim = ax_m.get_xlim()
        ylim = ax_m.get_ylim()
        ends = np.array(xlim)
        for means in self.mean:
            r = stats.linregress(lims, means)
            fitted_line = r.slope*ends + r.intercept
            ax_m.plot(ends, fitted_line, color='k', lw=0.5, zorder=1)
        #
        # Keep the x and y limits where they were prior to plotting
        # the fitted lines.
        ax_m.set_xlim(*xlim)
        ax_m.set_ylim(*ylim)
        #
        # Add title, axis labels, and legends.
        ax_m.xaxis.set_major_locator(plt.MaxNLocator(len(lims)))
        t = 'Linear Relations of Fitness Limit and Equilibrium Moments\n'
        t = t + 'Infinite Population, Beneficial Effects Weight $\gamma$'
        ax_m.set_title(t, size='large', pad=12)
        ax_m.set_xlabel('Upper Limit on Fitness')
        ax_m.set_ylabel('Mean Fitness at Equilibrium')
        ax_v.set_ylabel('Variance in Fitness at Equilibrium')
        ax_m.legend(title='Mean', loc='upper left')
        ax_v.legend(title='Variance', loc='lower right')
        #
        self.fig, self.ax_m, self.ax_v = fig, ax_m, ax_v
    
    def _begin_plot_curves(self, eq):
        # Set up an axis system and a color palette for a plot of
        # equlibria in one row or one column of the array.
        sns.set_palette(sns.color_palette("Set2", len(eq)))
        self.fig, self.ax = plt.subplots()
        self.lines = np.empty(len(eq), dtype=object)
        max_y = max([e.max() for e in eq])
        self.ax.set_ylim([-0.0003, 1.05 * max_y])
        
    def plot_column(self, j):
        """
        Plot equilibria for `j`-th upper limit on fitness.
        """
        eq = self.eq[:, j]
        fitness = self.params[0,j].m
        variable_name = 'Beneficial Effects Weight'
        abbrev_var_name = 'Weight $\gamma$'
        constant_name = 'Fitness Upper Limit {}'.format(fitness[-1])
        self._begin_plot_curves(eq)
        for i in range(len(eq)):
            label = exp_latex(self.mixture_weights[i])
            self.lines[i], = plt.plot(fitness, eq[i], label=label)
        self._finish_plot_curves(variable_name, abbrev_var_name,
                                 constant_name)

    def plot_row(self, i):
        """
        Plot equilibria for `i`-th weighting of beneficial effects.
        """
        eq = self.eq[i, :]
        params = self.params[i, :]
        variable_name = 'Fitness Upper Limit'
        abbrev_var_name = 'Fitness Limit'
        weight = exp_latex(self.mixture_weights[i])
        constant_name = 'Beneficial Effects Weight {}'.format(weight)
        self._begin_plot_curves(eq)
        for j in range(len(eq)):
            label = '{:5.3f}'.format(self.fitness_limits[j])
            self.lines[j], = plt.plot(params[j].m, eq[j], label=label)
            self.lines[j].set_zorder(100 - j)
        self._finish_plot_curves(variable_name, abbrev_var_name,
                                 constant_name)

    def _finish_plot_curves(self, variable_name, abbrev_var_name,
                                  constant_name):
        # Set titles and label axes.
        self.ax.set_ylabel('Relative Frequency')
        self.ax.set_xlabel('Fitness')
        title = 'Dependence of Equilibrium on {}\nInfinite Population, {}'
        title = title.format(variable_name, constant_name)
        self.ax.set_title(title, size='large', pad=12)
        self.ax.legend(title=abbrev_var_name, loc='best')
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    def save_and_display(self, filename):
        """
        Saves the figure to disk, displays it, and closes the plot.
        """
        save_and_display(self.fig, filename)

    def __getitem__(self, key):
        """
        Returns an equilibrium along with mean and variance.
        """
        return self.eq[key], self.mean[key], self.var[key]
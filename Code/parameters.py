class Parameters(object):
    """
    Stores all parameter settings of the infinite-population model.    
    
    There are correspondingly named members for model parameters b, d,
    m, w, gamma, n, and W. Instance method `q()` returns the
    probability distribution over mutational effects, and class method
    `f()` returns the corresponding matrix.
    
    All numbers, other than integer `n`, are multiprecision floats.
    """
    def __init__(self, b_max=0.25, d='0.1', w='5e-4', gamma='1e-3'):
        """
        Sets all parameters of the infinite-population model.
        
        Parameters are derived from the maximum birth parameter 
        `b_max`, the death parameter `d`, the bin width `w`, and the
        weighting `gamma` of beneficial mutational effects. The number
        `n` of classes of organism is set to the integer nearest to 
        `b_max/w + 1`.
        
        The default parameter settings come from Section 5 of Basener
        and Sanford.
        """
        # Use `mp_float()` for explicit conversion to multiprecision
        # float. In mixed-mode operations, (arrays of) integers and
        # floats are promoted automatically to (arrays of) multi-
        # precision floats.
        self.d = mp_float(d)
        self.w = mp_float(w)
        self.gamma = mp_float(gamma)
        #
        # The number of classes is 1 plus the integer nearest to
        # `b_max / w`.
        self.n = int(mp.nint(b_max / self.w)) + 1
        #
        # Create an array of n evenly spaced birth parameters ranging
        # from 0 to (n-1)w.
        self.b = linspace(0, (self.n - 1) * self.w, self.n)
        #
        # Subtract the scalar death parameter `d` from all elements of
        # the array of birth parameters to get the array of fitnesses.
        self.m = self.b - self.d
        #
        # To obtain the derivative operator `W`, first scale the
        # columns of the matrix `f()` (probabilities of mutational
        # effects) by the corresponding birth parameters, and then
        # subtract the death parameter from the elements of the main
        # diagonal.
        self.W = self.f(self.q(self.gamma)) * self.b
        self.W[np.diag_indices(self.n)] -= self.d
        
    @classmethod
    def f(self, q):
        """
        Returns the matrix of probabilities of mutational effects.
        
        The given probability distribution (1-D array) `q` contains
        2n - 1 non-negative numbers summing to 1. 
        
        The returned matrix `f` is n-by-n. Each off-diagonal element
        `f[i,j]` is set to `q[n-1+i-j]`. Each element `f[j,j]` is set
        to make the sum of elements in column `j` equal to 1.
        """
        n = len(q) // 2 + 1
        f = np.empty((n, n), dtype=type(q[0]))
        for i in range(n):
            # Assign array `[q[i], q[i+1], ..., q[i+n-1]]` to column
            # `j=n-i-1` of the n-by-n matrix `f`. Then set `f[j,j]`
            # to make the sum of elements in column `j` equal to one.
            j = (n - 1) - i
            f[:,j] = q[i:i+n]
            f[j,j] += 1 - fsum(f[:,j])
        return f
    
    def q(self, gamma, beta=500):
        """
        Returns a probability distribution over mutational effects.
        
        Writing w for `self.w`, the bin width, and n for `self.n`,
        the number of classes of organism, the mutational effects
        are 
        
            (1-n)w, ..., -w, 0, w, ..., (n-1)w.
        
        The unnormalized probability of mutational effect x is 
        
            G(x + w/2) - G(x - w/2),
        
        where G is the cumulative distribution function of the
        weighted mixture of a Gamma distribution (shape alpha=0.5,
        rate `beta`) and its reflection. The weighting of the Gamma
        distribution in the mixture is `gamma`, a number in the
        closed interval [0, 1]. The default value of `beta` comes
        from Section 5 of Basener and Sanford.
        
        The returned probability distribution (array) is normalized.
        """
        # The walls of the width-w bins centered on the n - 1 positive
        # mutational effects are n evenly spaced points ranging from
        # (1-1/2)w to (n-1/2)w. Create array [1/2, 3/2, ..., n-1/2],
        # and then scale all of its elements by w.
        walls = self.w * linspace(1/2, self.n - 1/2, self.n)
        
        # Calculate bin masses by differencing values of the comple-
        # mentary CDF of the Gamma distribution at the bin walls. 
        # The slice [:-1] of an array includes all elements but the
        # last, and the slice [1:] includes all elements but the first.
        # Note that alpha=0.5 is a special case. 
        complementary_cdf = erfc((beta * walls) ** 0.5)
        masses = complementary_cdf[:-1] - complementary_cdf[1:]
        #
        # The mass of the bin centered on zero is the value of the 
        # Gamma CDF at w/2 == walls[0].
        zero_mass = 1 - complementary_cdf[0]
        #
        # The upper (lower) tail of the binned mixture is obtained by
        # weighting the (reversed) `masses`.
        upper_tail = gamma * masses
        lower_tail = (1 - gamma) * masses[::-1]
        #
        # Assemble the pieces into a single array, and normalize.
        q = np.concatenate((lower_tail, [zero_mass], upper_tail))
        return q / fsum(q)

    def initial_freqs(self, mean='0.044', std='0.005'):
        """
        Returns a binned normal distribution over fitness.
        
        The unnormalized probability of fitness `m[i]` is
        
            `F(m[i] + self.w/2) - F(m[i] - self.w/2)`,
            
        where `F` is the cumulative distribution function of the
        normal distribution with the given `mean` and standard
        deviation `std`. The default settings of the parameters come
        from Section 5 of Basener and Sanford.
        
        The returned probability distribution (array) is normalized.
        """
        mean = mp_float(mean)
        std = mp_float(std)
        #
        # The n fitnesses are 0w - d, 1w - d, ..., (n-1)w - d. The
        # n + 1 walls of the width-w bins centered on the fitnesses
        # are (0-1/2)w - d, (1-1/2)w - d, ..., (n-1/2)w - d. Create 
        # an array [-1/2, 1/2, ..., n-1/2], scale all of the elements
        # by w, and then subtract d from all of the elements.
        walls = self.w * linspace(-1/2, self.n-1/2, self.n+1) - self.d
        #
        # Evaluate the CDF and the complementary CDF at the bin walls.
        z = (walls - mean) / (std * mp_float(2.0)**0.5)
        cdf = 0.5 * (1 + erf(z))
        ccdf = 0.5 * erfc(z)
        #
        # Calculate bin masses by differencing CDF values at the bin
        # walls, and also by differencing complementary CDF values at
        # the bin walls. The slice [1:] includes all array elements
        # except the first, and the slice [:-1] includes all array
        # elements except the last.
        per_cdf = cdf[1:] - cdf[:-1]
        per_ccdf = ccdf[:-1] - ccdf[1:]
        #
        # For accuracy, use the CDF differences for bins with upper
        # walls no greater than the mean, and the complementary CDF
        # differences for other bins.
        freqs = np.where(walls[1:] <= mean, per_cdf, per_ccdf)
        return freqs / fsum(freqs)
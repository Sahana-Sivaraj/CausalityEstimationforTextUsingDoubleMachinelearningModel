class ReviewSimulation():
    @staticmethod
    def estimate_propensities(self,T, C):
        # estimate treatment distribution for each strata of the confound
        # directly from the data
        df = pd.DataFrame(zip(C, T), columns=['C', 'T'])
        T_levels = set(T)
        propensities = []
        for c_level in set(C):
            subset = df.loc[df.C == c_level]
            # NOTE: subset.T => transpose
            p_TgivenC = [
                float(len(subset.loc[subset['T'] == t])) / len(subset)
                for t in T_levels
            ]
            propensities.append(p_TgivenC[1])

        return propensities

    # b0  makes treatment (thm?) sepearte more (i.e. give more 1's)
    # b1 1, 10, 100, makes confound (buzzy/not) seperate more (drives means apart)
    # gamma 0 , 1, 4, noise level
    # offset moves propensities towards the middle so sigmoid can split them into some noise
    @staticmethod
    def simulate_Y(self,C, T, b0=0.5, b1=10, gamma=0.0, offset=0.75):
        propensities = estimate_propensities(T, C)
        # propensities = [0.27, 0.7]
        out = []
        test = defaultdict(list)
        for Ci, Ti in zip(C, T):
            noise = np.random.normal(0, 1)
            y0 = b1 * (propensities[Ci] - offset)
            y1 = b0 + y0
            y = (1 - Ti) * y0 + Ti * y1 + gamma * noise  # gamma
            simulated_prob = ReviewSimulation.sigmoid(y)
            y0 = ReviewSimulation.sigmoid(y0)
            y1 = ReviewSimulation.sigmoid(y1)
            threshold = np.random.uniform(0, 1)
            Y = int(simulated_prob > threshold)
            out.append(Y)
            test[Ci, Ti].append(Y)

        return out

    @staticmethod
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def treatment_from_rating(self,rating):
        return int(rating == 5.0)

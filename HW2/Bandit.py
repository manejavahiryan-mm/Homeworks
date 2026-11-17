
from abc import ABC, abstractmethod
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Produce diagnostic plots without opening interactive windows."""

    def __init__(self, output_dir='plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot1(self, algorithms):
        """Bandit estimated means over time, on linear and log x-axes."""
        for algorithm in algorithms:
            for scale in ('linear', 'log'):
                fig, ax = plt.subplots(figsize=(10, 6))
                trials = np.arange(1, len(algorithm.rewards) + 1)
                for j, true_mean in enumerate(algorithm.means):
                    ax.plot(trials, algorithm.estimate_history[:, j], label=f'Arm {j+1} (true mean {true_mean:g})')
                ax.set_xscale(scale)
                ax.set_xlabel('Trial')
                ax.set_ylabel('Estimated mean reward')
                ax.set_title(f'{algorithm.name}: learning curves ({scale} scale)')
                ax.legend()
                ax.grid(alpha=0.2)
                fig.tight_layout()
                path = self.output_dir / f'{algorithm.slug}_learning_{scale}.png'
                fig.savefig(path, dpi=160)
                plt.close(fig)
                logger.info('Saved {}', path)

    def plot2(self, algorithms):
        """Plot observed cumulative rewards and expected cumulative regrets."""
        for metric, ylabel in (('cumulative_rewards', 'Cumulative observed reward'),
                              ('cumulative_regrets', 'Cumulative pseudo-regret')):
            fig, ax = plt.subplots(figsize=(10, 6))
            for algorithm in algorithms:
                ax.plot(np.arange(1, len(algorithm.rewards)+1), getattr(algorithm, metric), label=algorithm.name)
            ax.set_xlabel('Trial')
            ax.set_ylabel(ylabel)
            ax.set_title('Epsilon-Greedy vs. Thompson Sampling: ' + ylabel)
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            path = self.output_dir / f'comparison_{metric}.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            logger.info('Saved {}', path)

#--------------------------------------#


class _GaussianAlgorithm(Bandit):
    """Shared Gaussian-arm environment, logging and common metrics.

    Each arm returns N(true_mean, 1 / reward_precision). The algorithm's
    selection rule is provided by subclasses. The `p` argument holds means.
    """

    name = 'Base algorithm'
    slug = 'base'

    def __init__(self, p, n_trials=20000, seed=42, reward_precision=1.0):
        self.means = np.asarray(p, dtype=float)
        if self.means.ndim != 1 or len(self.means) < 2:
            raise ValueError('Provide at least two arm means')
        if n_trials <= 0 or reward_precision <= 0:
            raise ValueError('Trials and reward precision must be positive')
        self.n_trials = int(n_trials)
        self.reward_precision = float(reward_precision)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(len(self.means), dtype=int)
        self.estimates = np.zeros(len(self.means), dtype=float)
        self.rewards = np.empty(self.n_trials)
        self.chosen = np.empty(self.n_trials, dtype=int)
        self.regrets = np.empty(self.n_trials)
        self.estimate_history = np.empty((self.n_trials, len(self.means)))
        self.completed = False

    def __repr__(self):
        return f'{self.name}(means={self.means.tolist()}, trials={self.n_trials})'

    def pull(self, bandit_index):
        """Draw one normally distributed reward from the selected arm."""
        return float(self.rng.normal(self.means[bandit_index], 1 / np.sqrt(self.reward_precision)))

    def update(self, bandit_index, reward):
        """Online sample-mean update; subclasses may add posterior updates."""
        self.counts[bandit_index] += 1
        n = self.counts[bandit_index]
        self.estimates[bandit_index] += (reward - self.estimates[bandit_index]) / n

    def choose_arm(self, t):
        raise NotImplementedError

    def experiment(self):
        """Run one reproducible sequential multi-armed bandit experiment."""
        optimal_mean = float(self.means.max())
        for i in range(self.n_trials):
            arm = int(self.choose_arm(i + 1))
            reward = self.pull(arm)
            self.update(arm, reward)
            self.chosen[i] = arm
            self.rewards[i] = reward
            # Pseudo-regret uses expected arm means, not noisy realized rewards.
            self.regrets[i] = optimal_mean - self.means[arm]
            self.estimate_history[i] = self.estimates
        self.cumulative_rewards = np.cumsum(self.rewards)
        self.cumulative_regrets = np.cumsum(self.regrets)
        self.completed = True
        return self

    def report(self):
        """Log requested averages/totals and return structured metrics."""
        if not self.completed:
            raise RuntimeError('Call experiment() before report()')
        summary = {
            'algorithm': self.name,
            'trials': self.n_trials,
            'total_reward': float(self.cumulative_rewards[-1]),
            'average_reward': float(self.rewards.mean()),
            'cumulative_regret': float(self.cumulative_regrets[-1]),
            'average_regret': float(self.regrets.mean()),
            'optimal_arm_selection_rate': float(np.mean(self.chosen == int(np.argmax(self.means)))),
            'selection_counts': self.counts.tolist(),
            'estimated_means': self.estimates.tolist(),
        }
        logger.info('{} | total reward={:.2f} | average reward={:.4f} | cumulative regret={:.2f} | average regret={:.4f} | optimal selection={:.2%}',
                    self.name, summary['total_reward'], summary['average_reward'],
                    summary['cumulative_regret'], summary['average_regret'], summary['optimal_arm_selection_rate'])
        logger.info('{} | pulls per arm={} | estimated means={}', self.name, summary['selection_counts'],
                    [round(x, 3) for x in summary['estimated_means']])
        return summary


class EpsilonGreedy(_GaussianAlgorithm):
    """Explore with epsilon_t = min(1, epsilon_scale / t)."""
    name = 'Epsilon-Greedy'
    slug = 'epsilon_greedy'

    def __init__(self, p, n_trials=20000, seed=42, reward_precision=1.0, epsilon_scale=1.0):
        super().__init__(p, n_trials, seed, reward_precision)
        if epsilon_scale <= 0:
            raise ValueError('epsilon_scale must be positive')
        self.epsilon_scale = float(epsilon_scale)

    def choose_arm(self, t):
        epsilon = min(1.0, self.epsilon_scale / t)
        if self.rng.random() < epsilon:
            return self.rng.integers(len(self.means))
        # Random tie-breaking avoids always privileging arm 1.
        best = np.flatnonzero(self.estimates == self.estimates.max())
        return self.rng.choice(best)


class ThompsonSampling(_GaussianAlgorithm):
    """Normal-normal Thompson Sampling with known observation precision.

    Prior for arm i: mu_i ~ N(prior_mean, 1/prior_precision).
    Likelihood: reward | mu_i ~ N(mu_i, 1/reward_precision).
    Posterior precisions and means are updated analytically.
    """
    name = 'Thompson Sampling'
    slug = 'thompson_sampling'

    def __init__(self, p, n_trials=20000, seed=43, reward_precision=1.0,
                 prior_mean=0.0, prior_precision=1.0):
        super().__init__(p, n_trials, seed, reward_precision)
        if prior_precision <= 0:
            raise ValueError('Prior precision must be positive')
        self.posterior_means = np.full(len(self.means), float(prior_mean))
        self.posterior_precisions = np.full(len(self.means), float(prior_precision))

    def choose_arm(self, t):
        samples = self.rng.normal(self.posterior_means, 1 / np.sqrt(self.posterior_precisions))
        return int(np.argmax(samples))

    def update(self, bandit_index, reward):
        old_precision = self.posterior_precisions[bandit_index]
        new_precision = old_precision + self.reward_precision
        self.posterior_means[bandit_index] = (
            old_precision * self.posterior_means[bandit_index] + self.reward_precision * reward
        ) / new_precision
        self.posterior_precisions[bandit_index] = new_precision
        super().update(bandit_index, reward)


def comparison(means=(1, 2, 3, 4), n_trials=20000, seed=42, output_dir='.'):
    """Run both strategies and export plots, trial-level CSV and summary CSV.

    The two policies have independent, reproducible RNG streams. This is a
    single-run comparison; the README describes a stronger multi-seed study.
    """
    root = Path(output_dir)
    results_dir = root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    algorithms = [EpsilonGreedy(means, n_trials=n_trials, seed=seed),
                  ThompsonSampling(means, n_trials=n_trials, seed=seed + 1)]
    summaries = []
    for algorithm in algorithms:
        algorithm.experiment()
        summaries.append(algorithm.report())

    rewards_file = results_dir / 'rewards.csv'
    with rewards_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Bandit', 'Reward', 'Algorithm', 'Trial', 'Regret', 'Cumulative_Reward', 'Cumulative_Regret'])
        for alg in algorithms:
            for i in range(n_trials):
                writer.writerow([int(alg.chosen[i]) + 1, float(alg.rewards[i]), alg.name,
                                 i + 1, float(alg.regrets[i]), float(alg.cumulative_rewards[i]),
                                 float(alg.cumulative_regrets[i])])
    with (results_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Trials', 'Total_Reward', 'Average_Reward', 'Cumulative_Regret',
                         'Average_Regret', 'Optimal_Arm_Selection_Rate'])
        for x in summaries:
            writer.writerow([x[k] for k in ('algorithm', 'trials', 'total_reward', 'average_reward',
                                          'cumulative_regret', 'average_regret', 'optimal_arm_selection_rate')])
    visualization = Visualization(root / 'plots')
    visualization.plot1(algorithms)
    visualization.plot2(algorithms)
    logger.success('Saved 40,000 trial rows, summary and six plots in {}', root)
    return algorithms, summaries


if __name__ == '__main__':
    comparison(output_dir=Path(__file__).resolve().parent)

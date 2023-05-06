import matplotlib.pyplot as plt
from deep_rl.plotter import Plotter
import logging
import json
import os


def setup_logger():
    """
    Set up a logging framework to log info, warning and error messages.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

logger = setup_logger()


def read_config(filename):
    """
    Read the config file and load data into a dictionary.

    :param filename: Name of the config file.
    :return: A dictionary containing the data from the config file.
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def plot_games(plotter, games, patterns, labels, agg, downsample, right_align, tag, root, interpolation, window):
    """
    Plot the specified games.

    :param plotter: The plotter object to use.
    :param games: The games to plot.
    :param patterns: The patterns to use.
    :param labels: The labels to use.
    :param agg: How to aggregate the data.
    :param downsample: The number of steps to downsample.
    :param right_align: Whether to right-align the plots.
    :param tag: The tag to use.
    :param root: The root directory to search for data.
    :param interpolation: The number of interpolation steps to use.
    :param window: The size of the window for rolling averages.
    """
    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg=agg,
                       downsample=downsample,
                       labels=labels,
                       right_align=right_align,
                       tag=tag,
                       root=root,
                       interpolation=interpolation,
                       window=window
                      )


def plot_games_from_config(plotter, config):
    """
    Plot the games specified in the config file.

    :param plotter: The plotter object to use.
    :param config: The configuration file.
    """
    for game_config in config['games']:
        games = game_config['names']
        patterns = game_config['patterns']
        labels = game_config['labels']
        agg = game_config.get('agg', 'mean')
        downsample = game_config.get('downsample', 0)
        right_align = game_config.get('right_align', False)
        tag = game_config['tag']
        root = game_config['root']
        interpolation = game_config.get('interpolation', 0)
        window = game_config.get('window', 0)

        plot_games(plotter, games, patterns, labels, agg, downsample, right_align, tag, root, interpolation, window)


def main():
    """
    Entry point of the application.
    """
    logger.info("Starting the plotter...")
    config = read_config('config.json')
    plotter = Plotter()
    os.makedirs('images', exist_ok=True)
    plot_games_from_config(plotter, config)
    plt.tight_layout()
    plt.savefig('images/plot.png', bbox_inches='tight')
    logger.info("Plotting completed successfully.")


if __name__ == '__main__':
    main()
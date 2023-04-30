import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(file_path):
    """
    Load search results from a json file.

    Args:
        file_path (str): The file path to the json file.

    Returns:
        A list of tuples representing the search results.
    """
    with open(file_path) as f:
        return json.load(f)

def find_best_params(results):
    """
    Find the best parameters based on the minimum WER.

    Args:
        results (list of tuples): The search results.

    Returns:
        A tuple representing the best parameters (alpha, beta, WER, CER).
    """
    return min(results, key=lambda x: x[2])

def plot_results(alpha, beta, wer):
    """
    Plot the search results as a 3D surface plot.

    Args:
        alpha (np.array): The possible values for alpha.
        beta (np.array): The possible values for beta.
        wer (np.array): The WER values for different combinations of alpha and beta.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(alpha, beta)
    surf = ax.plot_surface(X, Y, wer, cmap=plt.cm.rainbow, linewidth=0, antialiased=False)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('WER')
    ax.set_zlim(5., 101.)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select the best parameters based on the WER')
    parser.add_argument('--input-path', type=str, help='Output json file from search_lm_params')
    args = parser.parse_args()

    try:
        results = load_results(args.input_path)
        alpha, beta, *_ = list(zip(*results))
        alpha = np.array(sorted(list(set(alpha))))
        beta = np.array(sorted(list(set(beta))))
        wer = np.array([[r[2] for r in results if r[0]==a and r[1]==b][0] for a in alpha] for b in beta])
        plot_results(alpha, beta, wer)
        print("Alpha: %f \nBeta: %f \nWER: %f\nCER: %f" % find_best_params(results))
    except Exception as e:
        print('An error occurred:', e)
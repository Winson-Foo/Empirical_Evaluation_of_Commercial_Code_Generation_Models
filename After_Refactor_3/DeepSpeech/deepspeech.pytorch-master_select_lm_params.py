import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_file(file_path):
    with open(file_path) as f:
        return json.load(f)

def find_min_wer(results):
    return min(results, key=lambda x: x[2])

def generate_arrays(results):
    alpha, beta, wer, cer = list(zip(*results))
    alpha = np.array(sorted(list(set(alpha))))
    beta = np.array(sorted(list(set(beta))))
    X, Y = np.meshgrid(alpha, beta)
    results = {(a, b): (w, c) for a, b, w, c in results}
    WER = np.array([[results[(a, b)][0] for a in alpha] for b in beta])
    return X, Y, WER

def plot_chart(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=matplotlib.cm.rainbow,
        linewidth=0,
        antialiased=False
    )
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('WER')
    ax.set_zlim(5., 101.)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Select the best parameters based on the WER')
    parser.add_argument('--input-path', type=str, help='Output json file from search_lm_params')
    args = parser.parse_args()

    results = load_file(args.input_path)

    min_results = find_min_wer(results)
    print("Alpha: %f \nBeta: %f \nWER: %f\nCER: %f" % tuple(min_results))

    X, Y, WER = generate_arrays(results)

    plot_chart(X, Y, WER)

if __name__ == '__main__':
    main()
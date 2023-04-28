import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_results(file_path):
    with open(file_path) as f:
        return json.load(f)

def find_min_WER(results):
    return min(results, key=lambda x: x[2])

def plot_WER_surface(alpha, beta, results):
    alpha = np.array(sorted(list(set(alpha))))
    beta = np.array(sorted(list(set(beta))))
    X, Y = np.meshgrid(alpha, beta)
    results_dict = {(a, b): (w, c) for a, b, w, c in results}
    WER = np.array([[results_dict[(a, b)][0] for a in alpha] for b in beta])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        X,
        Y,
        WER,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select the best parameters based on the WER')
    parser.add_argument('--input-path', type=str, help='Output json file from search_lm_params')
    args = parser.parse_args()
    
    results = load_results(args.input_path)
    min_results = find_min_WER(results)
    print(f"Alpha: {min_results[0]} \nBeta: {min_results[1]} \nWER: {min_results[2]}\nCER: {min_results[3]}")
    
    alpha, beta, *_ = list(zip(*results))
    plot_WER_surface(alpha, beta, results)

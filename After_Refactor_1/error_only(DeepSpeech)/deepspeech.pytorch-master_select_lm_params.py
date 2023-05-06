import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Select the best parameters based on the WER')
parser.add_argument('--input-path', type=str, help='Output json file from search_lm_params')

try:
    args = parser.parse_args()
except:
    print("Error: Invalid arguments given")
    exit()

try:
    with open(args.input_path) as f:
        results = json.load(f)
except:
    print("Error: Invalid input path")
    exit()

if not results:
    print("Error: Empty results found")
    exit()

min_results = min(results, key=lambda x: x[2])  # Find the minimum WER (alpha, beta, WER, CER)
print("Alpha: %f \nBeta: %f \nWER: %f\nCER: %f" % tuple(min_results))

alpha, beta, *_ = list(zip(*results))
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

try:
    plt.show()
except:
    print("Error: Couldn't display the plot")
    exit()
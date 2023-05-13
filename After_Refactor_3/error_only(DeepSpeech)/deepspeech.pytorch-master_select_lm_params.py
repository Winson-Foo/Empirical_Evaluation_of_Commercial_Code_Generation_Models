import argparse
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Select the best parameters based on the WER')
parser.add_argument('--input-path', type=str, help='Output json file from search_lm_params')
args = parser.parse_args()

if not args.input_path:
    print("Error: Input path is required")
    sys.exit()

try:
    with open(args.input_path) as f:
        results = json.load(f)
except FileNotFoundError:
    print("Error: Input file not found")
    sys.exit()
except json.JSONDecodeError:
    print("Error: Input file is not valid json")
    sys.exit()

if not isinstance(results, list):
    print("Error: Results are not in the correct format")
    sys.exit()

if len(results) == 0:
    print("Error: Results are empty")
    sys.exit()

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
plt.show()
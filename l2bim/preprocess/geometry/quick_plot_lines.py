# quick_plot_lines.py
import numpy as np, matplotlib.pyplot as plt
L = np.loadtxt(r"F:\Google\Download\4F_Region3\4F_Region3\submap\data\out_lines\lines_000013.txt")
if L.ndim == 1: L = L[None, :]
for x1, y1, x2, y2 in L:
    plt.plot([x1, x2], [y1, y2], '-')
plt.axis('equal'); plt.grid(True); plt.show()

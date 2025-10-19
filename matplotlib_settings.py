import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# Optionally adjust font sizes
mpl.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


# Set global legend style (capturing seaborn-default look)
mpl.rcParams.update({
    "legend.frameon": True,
    "legend.facecolor": 'inherit',   # Inherit background for natural integration
    "legend.edgecolor": 'inherit',   # Same for edge color
    "legend.framealpha": 0.75         # Slight transparency like seaborn
})

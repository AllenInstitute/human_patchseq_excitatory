import matplotlib.pyplot as plt
import numpy as np 

# Custom autopct to only show values > 4%
def _autopct_filter(pct):
    return f'{pct:.1f}%' if pct > 4 else ''

def plot_abundance(df, ax, color_dict, label=True, radius=1, center=(0,0)):

    if label: labels = df['supertype']
    else: labels = None

    wedges, texts = ax.pie(df['supertype_frac_of_layer'], 
                        labels=labels, 
                        colors=color_dict, 
                        # autopct=_autopct_filter, 
                        startangle=90, 
                        counterclock=False, 
                        rotatelabels=True,
                        textprops={'fontsize': 4},
                        radius = radius,
                        center = center)
    
    # Add labels manually around the correct center
    angle = 90
    for w, label in zip(wedges, labels):
        theta = (w.theta2 + w.theta1) / 2
        x = center[0] + 1.4 * radius * np.cos(np.deg2rad(theta))
        y = center[1] + 1.4 * radius * np.sin(np.deg2rad(theta))
        ax.text(x, y, label, ha='center', va='center', fontsize=6)

    ax.set_aspect('equal')

    # Color label text
    for text, color in zip(texts, color_dict):
        text.set_color(color)

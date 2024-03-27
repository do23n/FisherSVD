import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(x_values, y_values, x_label, y_label, title, heatmap_matrix, filename):
    fig, ax = plt.subplots(figsize=(10,10))
    heatmap = ax.imshow(heatmap_matrix, cmap='viridis')

    # set labels and ticks
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticklabels(y_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Rotating the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Adding colorbar and title
    cbar = plt.colorbar(heatmap)
    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for i in range(len(heatmap_matrix)):
        for j in range(len(heatmap_matrix[0])):
            ax.text(j, i, round(heatmap_matrix[i, j],2),
                    ha="center", va="center", color="w")

    # Save the plot
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_sensitivity_heatmap(sensitivity_dict, num_adj_blocks, title, filename):
    layer_names = list(sensitivity_dict.keys())
    param_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    perplexity_values = np.array(
        [[sensitivity_dict[layer_name][ratio] for ratio in param_ratios] 
         for layer_name in layer_names]
         ).T

    # block_size = 7
    i = 0
    while i < len(layer_names):
        cur_layer = layer_names[i]
        cur_block_size = 7 if "." in cur_layer else 1
        next_block_size = 7
        print("i={}, layer_names[i]={}".format(i, layer_names[i]))

        layer_idx = int(layer_names[i].split(".")[2]) if "." in cur_layer else -1
        if  "." not in cur_layer:
            fname = filename + f"/adj_blocks_{num_adj_blocks}/layer-{cur_layer}"
        elif num_adj_blocks == 1:
            fname = filename + f"/adj_blocks_{num_adj_blocks}/layer-{layer_idx}"
        elif num_adj_blocks > 1:
            fname = filename + f"/adj_blocks_{num_adj_blocks}/layer-{layer_idx}-{layer_idx - (num_adj_blocks - 1)}"

        end_idx = i + cur_block_size + next_block_size*(num_adj_blocks-1)
        plot_heatmap(x_values=layer_names[i:end_idx], 
                    y_values=param_ratios, 
                    x_label="layer", y_label="truncation ratio", title=title, 
                    heatmap_matrix=perplexity_values[:,i:end_idx],
                    filename=fname)
        i += cur_block_size

def plot_lines(x_values, y1_values, y2_values, x_label, y1_label, y2_label, title, truncation_ratio, filename):
    fig, ax1 = plt.subplots(figsize=(10,10))

    color = 'tab:red'
    ax1.set_xlabel(x_label, fontsize=15)
    ax1.set_ylabel(y1_label, color=color, fontsize=15)
    line1 = ax1.plot(x_values, y1_values, "-o", color=color, label='sensitivity for trunc_ratio = {}'.format(truncation_ratio))
    ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.set_title(title, fontsize=15)

    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y2_label, color=color, fontsize=15) # we already handled the x-label with ax1
    line2 = ax2.plot(x_values, y2_values, "-o", color=color, label='mean fisher info')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=15)

    lines = line1 + line2
    ax1.legend(lines, [l.get_label() for l in lines], loc=1, prop={'size': 10})
    fig.tight_layout()

    # Rotating the x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Save the plot
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_sensitivity_and_fisher(sensitivity_dict, fisher_dict, truncation_ratio, num_adj_blocks, title, filename):
    layer_names = list(sensitivity_dict.keys())
    i = 0
    while i < len(layer_names):
        cur_layer = layer_names[i]
        cur_block_size = 7 if "." in cur_layer else 1
        next_block_size = 7

        layer_idx = int(layer_names[i].split(".")[2]) if "." in cur_layer else -1
        if  "." not in cur_layer:
            fname = filename + f"/adj_blocks_{num_adj_blocks}/layer-{cur_layer}"
        elif num_adj_blocks == 1:
            fname = filename + f"/adj_blocks_{num_adj_blocks}/layer-{layer_idx}"
        elif num_adj_blocks > 1:
            fname = filename + f"/adj_blocks_{num_adj_blocks}/layer-{layer_idx}-{layer_idx - (num_adj_blocks - 1)}"

        end_idx = i + cur_block_size + next_block_size*(num_adj_blocks-1)

        perplexity_values = [sensitivity_dict[layer_name][truncation_ratio] for layer_name in layer_names][i:end_idx]
        fisher_values = [fisher_dict[name] for name in layer_names[i:end_idx]]
        
        plot_lines(x_values=layer_names[i:end_idx], 
                   y1_values=perplexity_values, 
                   y2_values=fisher_values, 
                   x_label="layer", 
                   y1_label="sensitivity", y2_label='mean fisher', title=title, 
                   truncation_ratio = truncation_ratio,
                   filename=fname)

        i += cur_block_size
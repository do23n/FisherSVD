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

    block_size = 7
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

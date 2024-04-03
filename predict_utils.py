import numpy as np
from sklearn.linear_model import LinearRegression

def linearize_ppl_from_fisher(sensitivity_dict, fisher_dict):
    layer_names = list(sensitivity_dict.keys())
    layer_names.remove('full_model')
    num_layers = len(layer_names)
    fisher_values = [fisher_dict[name] for name in layer_names]

    print("linearize_ppl_from_fisher()")
    all_reg_score = list()
    for truncation_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        X = [list(pair) for pair in zip([truncation_ratio] * num_layers, fisher_values)]  # [trunc_ratio, mean fisher]
        Y = [sensitivity_dict[layer][truncation_ratio] for layer in layer_names]
        reg = LinearRegression().fit(X, Y)
        all_reg_score.append(reg.score(X,Y))
        print("ratio={}: reg.score(X,Y)={}".format(truncation_ratio, reg.score(X,Y)))
        
    return np.array(all_reg_score)

def linearize_ppl_across_truncation_ratio(sensitivity_dict):
    layer_names = list(sensitivity_dict.keys())
    layer_names.remove('full_model')
    truncation_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("linearize_ppl_across_truncation_ratio()")
    all_reg_score = list()
    for layer in layer_names:
        Y_ppl = np.array([sensitivity_dict[layer][ratio] for ratio in truncation_ratios]).reshape(-1,1)
        X_ratio = np.array(truncation_ratios).reshape(-1,1)
        reg = LinearRegression().fit(X_ratio, Y_ppl)
        all_reg_score.append(reg.score(X_ratio, Y_ppl))
        print("{}: reg.score(X,Y)={}".format(layer, reg.score(X_ratio, Y_ppl)))

    return np.array(all_reg_score)

    
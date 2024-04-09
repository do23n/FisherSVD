import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig
from m_evaluate import evaluate_model
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info, get_fisher_info_per_element
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from predict_utils import linearize_ppl_from_fisher, linearize_ppl_across_truncation_ratio
from quantization import rtn_quant_sequential
from binary_search import binary_search_truncation_rank
from plot_utils import plot_sensitivity_heatmap, plot_sensitivity_and_fisher, plot_reg_score_ppl_across_ratio, plot_reg_score_ppl_from_fisher, plot_final_ratio


def main(args):
    model_id = args.model_id

    # Load model
    if "Llama-2" in model_id:
        hf_auth = args.huggingface_token
        tokenizer = AutoTokenizer.from_pretrained(model_id,token=hf_auth, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,token=hf_auth, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )

    # if "llama" in model_id or "opt" in model_id:
    #     model = model.to_bettertransformer()
    
    # save weight matrix shape
    with open("output/weight_shape.txt", "r+") as f:
        if model_id + "\n" not in f.readlines():
            f.write(f"{model_id}:\n")
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    f.write(f"{name}: {module.weight.shape}")
        
    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, 256)

    # get mean of per-element fisher information for each layer
    mean_fisher_dict = get_fisher_info_per_element(model, calib_loader, args.use_cache)

    if "fisher" in args.scaling_method:
        calib_fisher_info(model, calib_loader, args.use_cache)
    if "abs" in args.scaling_method:
        calib_input_distribution(
            model, calib_loader, args.scaling_method, args.use_cache
        )
    if args.sensitivity_metric == "ppl":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
    elif args.sensitivity_metric == "stable_rank":
        sensitivity = calib_sensitivity_stable_rank(
            model, calib_loader, args, args.use_cache
        )

    all_reg_score_ppl_from_fisher = linearize_ppl_from_fisher(sensitivity_dict=sensitivity, fisher_dict=mean_fisher_dict)
    all_reg_score_ppl_across_ratio = linearize_ppl_across_truncation_ratio(sensitivity_dict=sensitivity)

    plot_reg_score_ppl_from_fisher(all_reg_score=all_reg_score_ppl_from_fisher,
                                  filename=f"figures/linearize_ppl_from_fisher", figsize=(10,10))

    plot_reg_score_ppl_across_ratio(sensitivity_dict=sensitivity, all_reg_score=all_reg_score_ppl_across_ratio,
                                    filename=f"figures/linearize_ppl_across_ratio", figsize=(50,30))

    plot_sensitivity_and_fisher(sensitivity_dict=sensitivity, fisher_dict=mean_fisher_dict,
                                truncation_ratio=0.9, num_adj_blocks=1, 
                                title="Layer Sensitivity and Mean Fisher Info for {}".format(model_id),
                                filename=f"figures/corr_ppl_fisher_{model_id.replace('/','_')}",
                                figsize=(10,10))

    plot_sensitivity_heatmap(sensitivity_dict=sensitivity, num_adj_blocks=2, 
                             title="Layer Sensitivity for {}".format(model_id), 
                             filename=f"figures/sensitivity_hm_{model_id.replace('/','_')}",
                             figsize=(10,10))    

    # search best truncation rank for each layer

    all_selected_ratio = binary_search_truncation_rank(model, sensitivity, calib_loader, args)

    plot_final_ratio(sensitivity_dict=sensitivity, all_selected_ratio=all_selected_ratio,
                     filename=f"figures/final_ratio_{model_id.replace('/','_')}",
                     figsize=(50,30))

    # quantization
    if args.weight_quant != "none":
        if args.weight_quant == "rtn_int8":
            rtn_quant_sequential(model, 8)
        elif args.weight_quant == "rtn_int6":
            rtn_quant_sequential(model, 6)

    # evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "mmlu" if args.eval_mmlu else "",
        eval_ppl="wikitext2,ptb",
        limit=-1,
    )
    print(result)
    with open("output/result.txt", "a+") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")

    # finished


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default="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",
        help="huggingface access token"
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    args = parser.parse_args()

    main(args)

"""
V3 训练配方（recipe）封装
=========================

将“这一版”的关键超参数组合封装成可复用预设，便于：
1) 一键复现稳定配置；
2) 保持 run.py 参数定义简洁；
3) 避免重复拼接长命令。
"""

V3_RECIPE_PRESETS = {
    # 达成主要目标的平衡版（T+1~T+3综合较优）
    "focus_best": {
        "epochs": 70,
        "early_stop_patience": 15,
        "v3_stage1_ratio": 0.20,
        "v3_stage2_ratio": 0.35,
        "v3_stage3_weights": "1.4,1.3,1.5,1.7,1.9",
        "v3_min_epochs_before_early_stop": 45,
        "v3_best_metric": "focus_rmse",
        "v3_focus_t1_weight": 0.5,
        "v3_focus_tail_weight": 1.0,
        "v3_long_steps": "3,4,5",
        "v3_long_distill_weight": 0.12,
    },
    # T+5 定向增强版
    "t5_boost": {
        "epochs": 80,
        "early_stop_patience": 18,
        "v3_stage1_ratio": 0.15,
        "v3_stage2_ratio": 0.45,
        "v3_stage2_weights": "0.9,1.2,1.8,2.4,3.0",
        "v3_stage3_weights": "1.0,1.1,1.5,2.1,3.2",
        "v3_min_epochs_before_early_stop": 55,
        "v3_best_metric": "focus_rmse",
        "v3_focus_t1_weight": 0.3,
        "v3_focus_tail_weight": 1.3,
        "v3_long_steps": "4,5",
        "v3_long_distill_weight": 0.18,
    },
    # Single-backbone self-distillation version (no V2 teacher).
    "self_distill": {
        "epochs": 80,
        "early_stop_patience": 18,
        "v3_stage1_ratio": 0.15,
        "v3_stage2_ratio": 0.45,
        "v3_stage2_weights": "0.9,1.2,1.8,2.4,3.0",
        "v3_stage3_weights": "1.0,1.1,1.5,2.1,3.2",
        "v3_min_epochs_before_early_stop": 55,
        "v3_best_metric": "focus_rmse",
        "v3_focus_t1_weight": 0.3,
        "v3_focus_tail_weight": 1.3,
        "v3_long_teacher_source": "self",
        "v3_long_steps": "4,5",
        "v3_long_distill_weight": 0.16,
    },
    # self_distill 的 T+5 定向最小调参版（仅改长步集合与蒸馏权重）
    "self_distill_t5": {
        "epochs": 80,
        "early_stop_patience": 18,
        "v3_stage1_ratio": 0.15,
        "v3_stage2_ratio": 0.45,
        "v3_stage2_weights": "0.9,1.2,1.8,2.4,3.0",
        "v3_stage3_weights": "1.0,1.1,1.5,2.1,3.2",
        "v3_min_epochs_before_early_stop": 55,
        "v3_best_metric": "focus_rmse",
        "v3_focus_t1_weight": 0.3,
        "v3_focus_tail_weight": 1.3,
        "v3_long_teacher_source": "self",
        "v3_long_steps": "5",
        "v3_long_distill_weight": 0.06,
    },
}


def apply_v3_recipe(args):
    """
    根据 args.v3_recipe 覆盖参数。
    仅在 model_name=TripleStreamV3 且 recipe!=none 时生效。
    """
    recipe_name = getattr(args, "v3_recipe", "none")
    if getattr(args, "model_name", None) != "TripleStreamV3":
        return
    if recipe_name == "none":
        return

    preset = V3_RECIPE_PRESETS.get(recipe_name)
    if preset is None:
        raise ValueError(f"未知 v3_recipe: {recipe_name}")

    for key, value in preset.items():
        setattr(args, key, value)

import torch
import argparse
import random
import numpy as np
import os
from datetime import datetime

# 导入训练测试函数
from exp.exp import model_train, model_test
from exp.v3_recipe import apply_v3_recipe

# 导入模型
from my_model.model_v1 import TripleStreamFusionNetwork, count_parameters
from my_model.model_v2 import TripleStreamFusionNetworkV2
from my_model.model_v3 import TripleStreamFusionNetworkV3
from baselines.ablate_v3_structures import (
    AblateV3NoPersistence,
    AblateV3NoMixGate,
    AblateV3NoBinAware,
    AblateV3NoAux,
    AblateV3NoStreamGate,
)
from baselines.baseline_dlinear import LinearBaseline
from baselines.baseline_mlp import MLPBaseline
from baselines.baseline_lstm import LSTMBaseline
from baselines.baseline_lightgbm import LightGBMBaseline
from baselines.baseline_patchtst import PatchTSTBaseline
from baselines.baseline_tide import TIDEBaseline
from baselines.baseline_cnn_transformer import CNNTransformerBaseline
from baselines.baseline_itransformer import ITransformerBaseline


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_float_list_arg(value, arg_name, expected_len=None):
    try:
        values = [float(x.strip()) for x in value.split(',')]
    except Exception as exc:
        raise ValueError(f"--{arg_name} 格式错误，应为逗号分隔浮点数") from exc
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"--{arg_name} 需要 {expected_len} 个值，当前为 {len(values)}")
    return values


def parse_int_list_arg(value, arg_name):
    try:
        return [int(x.strip()) for x in value.split(',')]
    except Exception as exc:
        raise ValueError(f"--{arg_name} 格式错误，应为逗号分隔整数") from exc


def is_a_ablation_model(model_name):
    return str(model_name).startswith('AblateV3')


def apply_default_recipe_for_a_ablation(args):
    """
    A类结构消融默认沿用 self_distill 的训练策略，确保结构消融公平。
    """
    if not is_a_ablation_model(getattr(args, 'model_name', '')):
        return
    original_model = args.model_name
    original_recipe = getattr(args, 'v3_recipe', 'none')

    args.model_name = 'TripleStreamV3'
    args.v3_recipe = 'self_distill'
    apply_v3_recipe(args)

    args.model_name = original_model
    args.v3_recipe = original_recipe


def main():
    # 规范模型命名（用于日志/输出目录/实验配置）
    model_registry = {
        'TripleStream': TripleStreamFusionNetwork,
        'TripleStreamV2': TripleStreamFusionNetworkV2,
        'TripleStreamV3': TripleStreamFusionNetworkV3,
        'AblateV3NoPersistence': AblateV3NoPersistence,
        'AblateV3NoMixGate': AblateV3NoMixGate,
        'AblateV3NoBinAware': AblateV3NoBinAware,
        'AblateV3NoAux': AblateV3NoAux,
        'AblateV3NoStreamGate': AblateV3NoStreamGate,
        'BaselineLinear': LinearBaseline,
        'BaselineDLinear': LinearBaseline,
        'BaselineMLP': MLPBaseline,
        'BaselineLSTM': LSTMBaseline,
        'BaselineLightGBM': LightGBMBaseline,
        'BaselinePatchTST': PatchTSTBaseline,
        'BaselineTiDE': TIDEBaseline,
        'BaselineCNNTransformer': CNNTransformerBaseline,
        'BaselineiTransformer': ITransformerBaseline,
    }
    
    # 参数解析
    help_examples = (
        "常见用法示例:\n"
        "  1) BaselineLightGBM\n"
        "     python run.py --model_name BaselineLightGBM --site_name W2127_Haichaoba\n"
    )
    parser = argparse.ArgumentParser(
        description='雨滴谱短临降水预报',
        epilog=help_examples,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 关键参数（建议优先设置）
    parser.add_argument('--model_name', type=str, default='TripleStreamV3',
                        choices=list(model_registry.keys()),
                        help='模型名称')
    parser.add_argument('--v3_recipe', type=str, default='t5_boost',
                        choices=[
                            'none', 'focus_best', 't5_boost', 'self_distill', 'self_distill_t5',
                        ],
                        help='V3训练配方预设')
    parser.add_argument('--site_name', type=str, default='W2127_Haichaoba',
                        choices=['W2127_Haichaoba', 'W2128_Haichaoyinsi', 'W2129_buligou'],
                        help='站点名称')
    parser.add_argument('--no_train', action='store_false', dest='train_flag', default=1,
                        help='不训练模型，仅测试（默认会训练模型）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='最大训练轮次')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')

    parser.add_argument('--out_path', type=str, default='out',
                        help='输出路径')
    parser.add_argument('--v3_best_metric', type=str, default='main_rmse',
                        choices=['val_loss', 'main_rmse', 'focus_rmse'],
                        help='V3最佳模型与早停监控指标：val_loss(阶段损失) / main_rmse / focus_rmse')
    parser.add_argument('--v3_focus_t1_weight', type=float, default=0.5,
                        help='focus_rmse中T+1步RMSE权重')
    parser.add_argument('--v3_focus_tail_weight', type=float, default=1.0,
                        help='focus_rmse中T+2~T+N均值RMSE权重')
    parser.add_argument('--v3_long_distill_weight', type=float, default=0.12,
                        help='V3阶段2长时距蒸馏权重')
    parser.add_argument('--v3_long_teacher_source', type=str, default='self',
                        choices=['self'],
                        help='V3长时距蒸馏teacher来源（固定）：self=V3 stage1自蒸馏')
    parser.add_argument('--v3_long_steps', type=str, default='3,4,5',
                        help='V3长时距蒸馏步长，格式: 3,4,5')     

    # 数据窗口与切分配置
    parser.add_argument('--data_dir', type=str, default='data/图像preview3_Gap15_Len30',
                        help='数据目录')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='输入序列长度（分钟）')
    parser.add_argument('--pred_len', type=int, default=5,
                        help='预测序列长度（分钟）')
    parser.add_argument('--stride', type=int, default=1,
                        help='滑动窗口步长')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')

    # 模型结构配置
    # 注：conc/vel/phys 输入维度由数据协议固定（32/32/5），在模型内部维护，不再暴露为CLI参数。
    parser.add_argument('--cnn_hidden', type=int, default=64,
                        help='CNN隐藏层维度')
    parser.add_argument('--cnn_output', type=int, default=32,
                        help='CNN输出维度')
    parser.add_argument('--lstm_hidden', type=int, default=64,
                        help='LSTM隐藏层维度')
    parser.add_argument('--decoder_hidden', type=int, default=128,
                        help='解码器隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比例')
    parser.add_argument('--mlp_hidden', type=int, default=256,
                        help='MLP基线隐藏层维度')
    
    # 训练过程配置
    parser.add_argument('--patience', type=int, default=5,
                        help='学习率调度patience')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='早停patience')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader工作进程数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=2025,
                        help='随机种子')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    
    # 基础损失参数
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'weighted_mse'],
                        help='损失函数类型: mse=普通MSE, weighted_mse=加权MSE')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='强降水阈值（mm/h），仅用于weighted_mse')

    # 输出约束配置
    parser.add_argument('--no_nonneg_constraint', action='store_false', dest='enforce_non_negative', default=1,
                        help='关闭统一非负输出约束（默认开启）')

    # 高级参数区（条件生效，通常由 v3_recipe 管理）
    parser.add_argument('--alpha', type=float, default=0.32,
                        help='损失权重系数；仅在 --loss_type weighted_mse 时生效')
    parser.add_argument('--aux_loss_weight', type=float, default=0.05,
                        help='辅助任务损失权重；仅 V2/V3 多任务训练时生效')
    parser.add_argument('--aux_weights', type=str, default='1.0,1.0,1.0,1.0',
                        help='辅助任务分量权重（Dm,LogNw,LWC,Z）；仅 V2/V3 多任务训练时生效')
    parser.add_argument('--main_step_weights', type=str, default='2.0,1.0,1.0,1.0,1.0',
                        help='V3主任务步长权重（T+1,T+2,...）；仅在关闭分阶段训练(--no_v3_stage_train)时作为主权重')
    parser.add_argument('--no_v3_stage_train', action='store_false', dest='v3_stage_train', default=True,
                        help='关闭V3分阶段训练（默认开启）')
    parser.add_argument('--v3_stage1_ratio', type=float, default=0.30,
                        help='V3阶段1占总epoch比例（专注T+1）；仅分阶段训练开启时生效')
    parser.add_argument('--v3_stage2_ratio', type=float, default=0.50,
                        help='V3阶段2占总epoch比例（专注T+2~T+5）；仅分阶段训练开启时生效')
    parser.add_argument('--v3_stage1_weights', type=str, default='3.0,0.8,0.6,0.5,0.4',
                        help='V3阶段1主任务步长权重；仅分阶段训练开启时生效')
    parser.add_argument('--v3_stage2_weights', type=str, default='1.0,1.4,1.8,2.0,2.2',
                        help='V3阶段2主任务步长权重；仅分阶段训练开启时生效')
    parser.add_argument('--v3_stage3_weights', type=str, default='1.6,1.2,1.2,1.2,1.2',
                        help='V3阶段3主任务步长权重；仅分阶段训练开启时生效')
    parser.add_argument('--v3_t1_distill_weight', type=float, default=0.10,
                        help='V3阶段2对T+1蒸馏约束权重；仅分阶段训练开启且进入stage2时生效')
    parser.add_argument('--v3_min_epochs_before_early_stop', type=int, default=0,
                        help='V3最少训练轮次后才允许早停；仅分阶段训练开启时生效（0=自动按stage配置）')


    args = parser.parse_args()

    # 先应用配方，再统一做参数解析与校验
    if args.model_name == 'TripleStreamV3':
        apply_v3_recipe(args)
    else:
        apply_default_recipe_for_a_ablation(args)

    # 解析列表型参数
    args.aux_dim_weights = parse_float_list_arg(
        args.aux_weights, 'aux_weights', expected_len=4
    )
    args.main_step_weight_list = parse_float_list_arg(
        args.main_step_weights, 'main_step_weights', expected_len=args.pred_len
    )
    args.v3_stage1_weight_list = parse_float_list_arg(
        args.v3_stage1_weights, 'v3_stage1_weights', expected_len=args.pred_len
    )
    args.v3_stage2_weight_list = parse_float_list_arg(
        args.v3_stage2_weights, 'v3_stage2_weights', expected_len=args.pred_len
    )
    args.v3_stage3_weight_list = parse_float_list_arg(
        args.v3_stage3_weights, 'v3_stage3_weights', expected_len=args.pred_len
    )
    for name, arr in [
        ("v3_stage1_weights", args.v3_stage1_weight_list),
        ("v3_stage2_weights", args.v3_stage2_weight_list),
        ("v3_stage3_weights", args.v3_stage3_weight_list),
    ]:
        if len(arr) != args.pred_len:
            raise ValueError(
                f"--{name} 数量需与 pred_len 一致，当前 pred_len={args.pred_len}, weights={len(arr)}"
            )
    args.v3_long_step_list = parse_int_list_arg(args.v3_long_steps, 'v3_long_steps')
    if any(s < 1 or s > args.pred_len for s in args.v3_long_step_list):
        raise ValueError(
            f"--v3_long_steps 必须在 [1, {args.pred_len}] 范围内，当前为 {args.v3_long_step_list}"
        )
    args.v3_long_step_idx = [s - 1 for s in args.v3_long_step_list]

    # 设置随机种子
    setup_seed(args.seed)
    print(f"随机种子已设置: {args.seed}")
    
    # 检查GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"发现 {device_count} 个CUDA设备")
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f'使用GPU: cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
        print('使用CPU')
    
    # 创建实验ID
    now = datetime.now()
    exp_id = now.strftime("%Y%m%d_%H%M%S")
    
    # 创建保存路径（A类消融放入站点子目录）
    exp_folder_name = f"{args.model_name}_{exp_id}"

    if is_a_ablation_model(args.model_name):
        save_path = os.path.join(args.out_path, args.site_name, '消融实验', 'A', exp_folder_name)
    else:
        save_path = os.path.join(args.out_path, args.site_name, exp_folder_name)
    os.makedirs(save_path, exist_ok=True)
    args.save_path = save_path
    
    print("\n" + "="*70)
    print(f"实验配置:")
    print(f"  站点: {args.site_name}")
    print(f"  模型: {args.model_name}")
    print(f"  输入长度: {args.seq_len} 分钟")
    print(f"  预测长度: {args.pred_len} 分钟")
    print(f"  批大小: {args.batch_size}")
    print(f"  最大轮次: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  非负输出约束: {'开启' if args.enforce_non_negative else '关闭'}")
    if is_a_ablation_model(args.model_name):
        print("  消融类别: A（模型结构消融）")
        print("  A类训练策略基线: self_distill")
    if args.model_name == 'TripleStreamV2':
        print(f"  V2辅助损失权重: {args.aux_loss_weight}")
        print(f"  V2辅助分量权重(Dm,LogNw,LWC,Z): {args.aux_dim_weights}")
    if args.model_name == 'TripleStreamV3' or is_a_ablation_model(args.model_name):
        print(f"  V3训练配方: {args.v3_recipe}")
        print(f"  V3辅助损失权重: {args.aux_loss_weight}")
        print(f"  V3辅助分量权重(Dm,LogNw,LWC,Z): {args.aux_dim_weights}")
        print(f"  V3主任务步长权重(T+1..T+{args.pred_len}): {args.main_step_weight_list}")
        print(f"  V3最佳模型监控指标: {args.v3_best_metric}")
        if args.v3_best_metric == 'focus_rmse':
            print(f"    focus权重: T+1={args.v3_focus_t1_weight}, Tail={args.v3_focus_tail_weight}")
        print(f"  V3分阶段训练: {'开启' if args.v3_stage_train else '关闭'}")
        if args.v3_stage_train:
            print(f"    阶段1权重: {args.v3_stage1_weight_list}")
            print(f"    阶段2权重: {args.v3_stage2_weight_list}")
            print(f"    阶段3权重: {args.v3_stage3_weight_list}")
            print(f"    阶段比例: stage1={args.v3_stage1_ratio}, stage2={args.v3_stage2_ratio}")
            print(f"    阶段2 T+1蒸馏权重: {args.v3_t1_distill_weight}")
            print(f"    阶段2长时距蒸馏权重: {args.v3_long_distill_weight}")
            print(f"    阶段2长时距teacher来源: {args.v3_long_teacher_source}")
            print(f"    阶段2长时距蒸馏步长: {args.v3_long_step_list}")
            print("    V3自蒸馏模式：不使用V2 teacher")
            print(f"    早停最小轮次: {args.v3_min_epochs_before_early_stop} (0=自动)")
    print(f"  保存路径: {save_path}")
    print("="*70)
    
    # 创建模型（只需传入args参数）
    model_class = model_registry[args.model_name]
    model = model_class(args=args)
    
    # 统计参数量（仅对PyTorch模型）
    if isinstance(model, torch.nn.Module):
        param_count = count_parameters(model)
        print(f"\n模型参数量: {param_count:,}")
    
    # 训练或测试
    # 所有模型（主模型和baseline）都通过 model_name 和 train_flag 统一控制
    if args.train_flag:
        model_train(model, device, args)
    else:
        # 仅测试模式（适用于所有模型）
        model_test(model, device, args)
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"结果已保存到: {save_path}")
    print("="*70)


if __name__ == '__main__':
    main()
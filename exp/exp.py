import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import pickle
import os
import copy

# 本地模块导入
from data_loader.dataset import (
    prepare_datasets, load_preprocessed_data, 
    extract_dsd_features, create_sliding_window_samples
)
from my_model.model_v1 import WeightedMSELoss
from my_model.model_v2 import TripleStreamFusionNetworkV2
from utils.metrics import compute_all_metrics, print_metrics
from utils.feature_engineering import apply_log_transform
from exp.reporting import write_timeline_metrics_report
from exp.plotting import (
    aggregate_event_predictions_for_step,
    plot_training_curves,
    plot_timeline_aggregated_predictions,
    plot_timeline_stepwise_predictions,
    plot_timeline_stepwise_panel,
)


def apply_output_constraint(predictions, enforce_non_negative=True):
    """统一输出约束，确保所有模型在同一口径下比较。"""
    if not enforce_non_negative:
        return predictions

    if isinstance(predictions, torch.Tensor):
        return torch.clamp(predictions, min=0.0)
    return np.clip(predictions, a_min=0.0, a_max=None)


def model_supports_aux_task(model):
    return bool(getattr(model, 'supports_aux_task', False))


def is_non_torch_baseline(model):
    return hasattr(model, 'fit_baseline') and hasattr(model, 'predict_baseline')


def is_v3_family(model_name):
    return model_name == 'TripleStreamV3' or str(model_name).startswith('AblateV3')


def should_enable_v3_long_distill(args):
    if not is_v3_family(args.model_name):
        return False
    if getattr(args, 'v3_long_distill_weight', 0.0) <= 0:
        return False

    teacher_source = getattr(args, 'v3_long_teacher_source', 'v2')
    if teacher_source == 'self':
        return True

    return bool(getattr(args, 'v3_v2_teacher_ckpt', ''))


def load_v3_long_teacher(args, device):
    """加载V2教师模型用于V3长时距蒸馏。"""
    ckpt_path = args.v3_v2_teacher_ckpt
    if not os.path.exists(ckpt_path):
        print("[V3-Stage2] 警告：V2教师路径无效，跳过长时距蒸馏")
        return None

    teacher = TripleStreamFusionNetworkV2(args=args).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[V3-Stage2] 已加载V2教师: {ckpt_path}")
    return teacher


def standardize_aux_labels_inplace(datasets, save_path):
    """
    基于训练集统计量对辅助标签(Dm/LogNw/LWC/Z)做标准化。
    仅作用于 train/val，测试阶段不依赖辅助标签。
    """
    train_ds = datasets.get('train')
    if train_ds is None or train_ds.aux_labels is None:
        return None, None

    aux_mean = train_ds.aux_labels.mean(dim=(0, 1), keepdim=True)
    aux_std = train_ds.aux_labels.std(dim=(0, 1), keepdim=True)
    aux_std = torch.where(aux_std < 1e-6, torch.ones_like(aux_std), aux_std)

    for split_name in ('train', 'val'):
        ds = datasets.get(split_name)
        if ds is None or ds.aux_labels is None:
            continue
        ds.aux_labels = (ds.aux_labels - aux_mean) / aux_std

    with open(os.path.join(save_path, 'aux_label_scaler.pkl'), 'wb') as f:
        pickle.dump(
            {
                'mean': aux_mean.cpu().numpy(),
                'std': aux_std.cpu().numpy(),
            },
            f
        )

    return aux_mean, aux_std


def weighted_aux_mse(aux_pred, aux_label, aux_weights):
    """按分量加权的辅助任务 MSE。"""
    per_dim_mse = ((aux_pred - aux_label) ** 2).mean(dim=(0, 1))
    return (per_dim_mse * aux_weights).sum() / (aux_weights.sum() + 1e-12)


def compute_main_loss(predictions, labels, args, criterion, step_weights=None):
    """主任务损失（v3可选步长加权）。"""
    use_step_weight = (
        step_weights is not None and
        is_v3_family(args.model_name)
    )
    if not use_step_weight:
        return criterion(predictions, labels)

    if args.loss_type == 'mse':
        per_step = ((predictions - labels) ** 2).mean(dim=0)
    elif args.loss_type == 'weighted_mse':
        intensity_weight = 1.0 + args.alpha * torch.clamp(labels - args.threshold, min=0.0)
        weighted_sq = intensity_weight * ((predictions - labels) ** 2)
        per_step = weighted_sq.mean(dim=0)
    else:
        raise ValueError(f"不支持的损失函数类型: {args.loss_type}")

    return (per_step * step_weights).sum() / (step_weights.sum() + 1e-12)


def get_v3_stage_config(args):
    total = max(1, int(args.epochs))
    stage1_end = max(1, int(total * args.v3_stage1_ratio))
    stage2_end = max(stage1_end + 1, int(total * (args.v3_stage1_ratio + args.v3_stage2_ratio)))
    stage2_end = min(stage2_end, total)

    return {
        'stage1_end': stage1_end,
        'stage2_end': stage2_end,
        'stage1_weights': args.v3_stage1_weight_list,
        'stage2_weights': args.v3_stage2_weight_list,
        'stage3_weights': args.v3_stage3_weight_list,
    }


def current_v3_stage(epoch_idx, stage_cfg):
    if epoch_idx < stage_cfg['stage1_end']:
        return 'stage1'
    if epoch_idx < stage_cfg['stage2_end']:
        return 'stage2'
    return 'stage3'


def model_train(model, device, args):
    """
    模型训练主函数
    
    Args:
        model: 模型实例
        device: 设备
        args: 参数配置
    """
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50)
    
    use_aux_task = model_supports_aux_task(model)

    # 加载数据
    datasets, scaler, _ = prepare_datasets(
        site_name=args.site_name,
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        include_aux_phys=use_aux_task
    )
    
    train_count = len(datasets['train']) if datasets['train'] is not None else 0
    val_count = len(datasets['val']) if datasets['val'] is not None else 0
    print(f"训练集样本数: {train_count}")
    print(f"验证集样本数: {val_count}")

    if use_aux_task:
        _, _ = standardize_aux_labels_inplace(datasets, args.save_path)
        print("辅助标签标准化: 已基于训练集统计量应用到 train/val")

    # 保存scaler（所有模型都需要，测试阶段会读取）
    scaler.save(os.path.join(args.save_path, 'scaler.pkl'))

    # 非PyTorch基线（如 LightGBM）走独立训练分支
    if is_non_torch_baseline(model):
        print(f"\n检测到非PyTorch基线：{args.model_name}，使用专用训练流程")
        if datasets['train'] is None:
            raise ValueError("训练数据为空，无法训练非PyTorch基线模型。")
        model.fit_baseline(datasets['train'], datasets['val'])
        model.save_baseline(os.path.join(args.save_path, 'baseline_model.pkl'))

        # 保存一个占位训练历史，避免下游流程读取失败
        history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        with open(os.path.join(args.save_path, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)

        print("\n开始测试...")
        model_test(model, device, args)
        return

    # 移动模型到设备
    model = model.to(device)
    aux_weight_tensor = (
        torch.tensor(args.aux_dim_weights, dtype=torch.float32, device=device)
        if use_aux_task else None
    )
    main_step_weight_tensor = (
        torch.tensor(args.main_step_weight_list, dtype=torch.float32, device=device)
        if is_v3_family(args.model_name) else None
    )
    stage_cfg = get_v3_stage_config(args) if (is_v3_family(args.model_name) and args.v3_stage_train) else None
    teacher_model = None
    long_horizon_teacher_v2 = None
    last_stage_name = None
    min_epochs_before_early_stop = int(getattr(args, 'v3_min_epochs_before_early_stop', 0))
    if stage_cfg is not None and min_epochs_before_early_stop <= 0:
        # 自动模式：至少进入stage3后再允许早停，避免只训练到stage1/2
        min_epochs_before_early_stop = stage_cfg['stage2_end'] + 1
    min_epochs_before_early_stop = max(1, min_epochs_before_early_stop)
    if stage_cfg is not None:
        print(
            f"[V3-Stage] 早停最小轮次限制: {min_epochs_before_early_stop} "
            f"(stage1_end={stage_cfg['stage1_end']}, stage2_end={stage_cfg['stage2_end']})"
        )

    # 需要将标准化统计量注入模型（如 Persistence）
    if hasattr(model, 'set_scaler_stats'):
        model.set_scaler_stats(scaler.phys_mean, scaler.phys_std)
    
    # 定义损失函数和优化器
    if args.loss_type == 'mse':
        criterion = nn.MSELoss()
        print(f"使用损失函数: MSE")
    elif args.loss_type == 'weighted_mse':
        criterion = WeightedMSELoss(threshold=args.threshold, alpha=args.alpha)
        print(f"使用损失函数: WeightedMSE (threshold={args.threshold}, alpha={args.alpha})")
    else:
        raise ValueError(f"不支持的损失函数类型: {args.loss_type}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        print("\n该模型无可训练参数，跳过梯度训练，直接保存并测试。")
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
            'val_loss': float('nan'),
        }, os.path.join(args.save_path, 'checkpoint_best.pth'))

        history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        with open(os.path.join(args.save_path, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)

        print("\n开始测试...")
        model_test(model, device, args)
        return

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience
    )

    if datasets['train'] is None or datasets['val'] is None:
        raise ValueError("训练或验证数据为空，无法进行PyTorch模型训练。请检查数据划分比例与事件数量。")

    # 创建DataLoader（仅对可训练PyTorch模型）
    train_loader = DataLoader(
        datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 早停/最优模型监控
    best_val_loss = float('inf')
    best_monitor_metric = float('inf')
    monitor_metric_name = 'val_loss'
    if is_v3_family(args.model_name):
        monitor_metric_name = getattr(args, 'v3_best_metric', 'main_rmse')
    patience_counter = 0
    
    # 训练历史
    train_loss_log = []
    val_loss_log = []
    lr_log = []
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()

        epoch_step_weights = main_step_weight_tensor
        stage_name = None
        if stage_cfg is not None:
            stage_name = current_v3_stage(epoch, stage_cfg)
            if stage_name == 'stage1':
                epoch_step_weights = torch.tensor(stage_cfg['stage1_weights'], dtype=torch.float32, device=device)
            elif stage_name == 'stage2':
                epoch_step_weights = torch.tensor(stage_cfg['stage2_weights'], dtype=torch.float32, device=device)
            else:
                epoch_step_weights = torch.tensor(stage_cfg['stage3_weights'], dtype=torch.float32, device=device)

            if stage_name != last_stage_name:
                print(f"[V3-Stage] 进入 {stage_name} (epoch {epoch+1}/{args.epochs})")
                if stage_name == 'stage2':
                    teacher_model = copy.deepcopy(model).to(device)
                    teacher_model.eval()
                    for p in teacher_model.parameters():
                        p.requires_grad = False
                    print("[V3-Stage] 已冻结阶段1 teacher 用于T+1蒸馏约束")
                    if should_enable_v3_long_distill(args):
                        if getattr(args, 'v3_long_teacher_source', 'v2') == 'self':
                            long_horizon_teacher_v2 = teacher_model
                            print("[V3-Stage2] 使用V3 stage1 teacher进行长时距自蒸馏")
                        else:
                            long_horizon_teacher_v2 = load_v3_long_teacher(args, device)
                last_stage_name = stage_name
        
        train_losses = []
        for batch in train_loader:
            conc = batch['conc'].to(device)
            vel = batch['vel'].to(device)
            phys = batch['phys'].to(device)
            labels = batch['label'].to(device)
            aux_label = batch['aux_label'].to(device) if 'aux_label' in batch else None
            
            optimizer.zero_grad()
            
            # 前向传播（统一输出约束 + 可选多任务）
            if use_aux_task:
                predictions, aux_pred = model(conc, vel, phys, return_aux=True)
            else:
                predictions = model(conc, vel, phys)
                aux_pred = None

            predictions = apply_output_constraint(
                predictions,
                enforce_non_negative=args.enforce_non_negative
            )
            main_loss = compute_main_loss(
                predictions, labels, args, criterion, epoch_step_weights
            )

            if aux_pred is not None and aux_label is not None:
                aux_loss = weighted_aux_mse(aux_pred, aux_label, aux_weight_tensor)
                loss = main_loss + args.aux_loss_weight * aux_loss
            else:
                loss = main_loss

            # 阶段2：对 T+1 增加防遗忘蒸馏（teacher 为阶段1结束模型）
            if stage_name == 'stage2' and teacher_model is not None:
                with torch.no_grad():
                    teacher_pred = teacher_model(conc, vel, phys)
                    teacher_pred = apply_output_constraint(
                        teacher_pred, enforce_non_negative=args.enforce_non_negative
                    )
                t1_distill = F.mse_loss(predictions[:, 0], teacher_pred[:, 0])
                loss = loss + args.v3_t1_distill_weight * t1_distill

            # V3 阶段2：蒸馏长时距步（teacher 可选 V2 / V3-self）
            if (
                should_enable_v3_long_distill(args) and
                stage_name == 'stage2' and
                long_horizon_teacher_v2 is not None
            ):
                with torch.no_grad():
                    long_teacher_pred = long_horizon_teacher_v2(conc, vel, phys)
                    long_teacher_pred = apply_output_constraint(
                        long_teacher_pred, enforce_non_negative=args.enforce_non_negative
                    )
                step_idx = args.v3_long_step_idx
                long_distill = F.mse_loss(predictions[:, step_idx], long_teacher_pred[:, step_idx])
                loss = loss + args.v3_long_distill_weight * long_distill
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 平均训练损失
        avg_train_loss = np.mean(train_losses)
        train_loss_log.append(avg_train_loss)
        
        # 验证
        val_stats = model_validate(
            model, val_loader, criterion, device, args,
            step_weight_override=epoch_step_weights,
            return_stats=True
        )
        avg_val_loss = val_stats['loss_total']
        val_main_rmse = val_stats['main_rmse']
        val_focus_rmse = val_stats.get('focus_rmse', val_main_rmse)
        val_loss_log.append(avg_val_loss)
        
        # 学习率调度
        scheduler_signal = avg_val_loss
        if monitor_metric_name == 'main_rmse':
            scheduler_signal = val_main_rmse
        elif monitor_metric_name == 'focus_rmse':
            scheduler_signal = val_focus_rmse
        scheduler.step(scheduler_signal)
        current_lr = optimizer.param_groups[0]['lr']
        lr_log.append(current_lr)
        
        # 计算耗时
        elapsed = time.time() - start_time
        
        # 打印信息
        if is_v3_family(args.model_name):
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Val MainRMSE: {val_main_rmse:.6f} | "
                  f"Val FocusRMSE: {val_focus_rmse:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.0f}s")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.0f}s")
        
        # 保存最佳模型
        if monitor_metric_name == 'main_rmse':
            current_monitor_metric = val_main_rmse
        elif monitor_metric_name == 'focus_rmse':
            current_monitor_metric = val_focus_rmse
        else:
            current_monitor_metric = avg_val_loss

        if current_monitor_metric < best_monitor_metric:
            best_monitor_metric = current_monitor_metric
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'monitor_metric_name': monitor_metric_name,
                'monitor_metric_value': current_monitor_metric,
            }, os.path.join(args.save_path, 'checkpoint_best.pth'))
            print(
                f"  [best] 最佳模型已保存 "
                f"({monitor_metric_name}={current_monitor_metric:.6f}, Val Loss={avg_val_loss:.6f})"
            )
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= args.early_stop_patience:
            if (epoch + 1) >= min_epochs_before_early_stop:
                print(
                    f"\n早停触发！监控指标({monitor_metric_name}) "
                    f"{args.early_stop_patience} 个epoch未改善"
                )
                break
            print(
                f"  [early-stop hold] patience已满足，但当前epoch={epoch+1} < "
                f"min_epochs={min_epochs_before_early_stop}，继续训练"
            )
    
    # 保存训练历史
    history = {
        'train_loss': train_loss_log,
        'val_loss': val_loss_log,
        'learning_rate': lr_log
    }
    
    with open(os.path.join(args.save_path, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 保存scaler
    scaler.save(os.path.join(args.save_path, 'scaler.pkl'))
    
    print("\n" + "="*50)
    print(
        f"训练完成！最佳监控指标({monitor_metric_name}): {best_monitor_metric:.6f}, "
        f"对应Val Loss: {best_val_loss:.6f}"
    )
    print("="*50)
    
    # 绘制训练曲线
    plot_training_curves(history, args.save_path)
    
    # 训练结束后自动进行测试
    print("\n开始测试...")
    model_test(model, device, args)


def model_validate(model, val_loader, criterion, device, args, step_weight_override=None, return_stats=False):
    """
    模型验证函数
    
    Args:
        model: 模型实例
        val_loader: 验证集DataLoader
        criterion: 损失函数
        device: 设备
        args: 参数配置
        
    Returns:
        avg_loss: 平均验证损失
    """
    model.eval()
    val_losses = []
    use_aux_task = model_supports_aux_task(model)
    aux_weight_tensor = (
        torch.tensor(args.aux_dim_weights, dtype=torch.float32, device=device)
        if use_aux_task else None
    )
    main_step_weight_tensor = step_weight_override if step_weight_override is not None else (
        torch.tensor(args.main_step_weight_list, dtype=torch.float32, device=device)
        if is_v3_family(args.model_name) else None
    )
    
    main_sq_sum = 0.0
    main_abs_sum = 0.0
    main_count = 0
    step_sq_sum = np.zeros(args.pred_len, dtype=np.float64)
    step_count = np.zeros(args.pred_len, dtype=np.float64)

    with torch.no_grad():
        for batch in val_loader:
            conc = batch['conc'].to(device)
            vel = batch['vel'].to(device)
            phys = batch['phys'].to(device)
            labels = batch['label'].to(device)
            aux_label = batch['aux_label'].to(device) if 'aux_label' in batch else None
            
            if use_aux_task:
                predictions, aux_pred = model(conc, vel, phys, return_aux=True)
            else:
                predictions = model(conc, vel, phys)
                aux_pred = None

            predictions = apply_output_constraint(
                predictions,
                enforce_non_negative=args.enforce_non_negative
            )
            main_loss = compute_main_loss(
                predictions, labels, args, criterion, main_step_weight_tensor
            )

            if aux_pred is not None and aux_label is not None:
                aux_loss = weighted_aux_mse(aux_pred, aux_label, aux_weight_tensor)
                loss = main_loss + args.aux_loss_weight * aux_loss
            else:
                loss = main_loss
            
            val_losses.append(loss.item())

            err = predictions - labels
            main_sq_sum += torch.sum(err ** 2).item()
            main_abs_sum += torch.sum(torch.abs(err)).item()
            main_count += labels.numel()
            for step_idx in range(min(args.pred_len, labels.shape[1])):
                step_err = err[:, step_idx]
                step_sq_sum[step_idx] += torch.sum(step_err ** 2).item()
                step_count[step_idx] += step_err.numel()

    avg_loss = np.mean(val_losses)
    main_rmse = float(np.sqrt(main_sq_sum / (main_count + 1e-12)))
    main_mae = float(main_abs_sum / (main_count + 1e-12))
    step_rmse = [
        float(np.sqrt(step_sq_sum[i] / (step_count[i] + 1e-12)))
        for i in range(args.pred_len)
    ]

    focus_t1_weight = float(getattr(args, 'v3_focus_t1_weight', 0.5))
    focus_tail_weight = float(getattr(args, 'v3_focus_tail_weight', 1.0))
    if args.pred_len > 1:
        tail_mean_rmse = float(np.mean(step_rmse[1:]))
    else:
        tail_mean_rmse = step_rmse[0]
    focus_rmse = focus_t1_weight * step_rmse[0] + focus_tail_weight * tail_mean_rmse

    if return_stats:
        return {
            'loss_total': avg_loss,
            'main_rmse': main_rmse,
            'main_mae': main_mae,
            'step_rmse': step_rmse,
            'focus_rmse': float(focus_rmse),
        }
    return avg_loss


def aggregate_event_predictions_to_timeline(preds_event, labels_event, event_len, seq_len, pred_len):
    """
    将滑窗多步预测聚合到唯一时间点，避免整体评估中同一时刻被重复统计。

    Args:
        preds_event: (N, pred_len) 事件内窗口预测
        labels_event: (N, pred_len) 事件内窗口标签
        event_len: 事件总时长（分钟数）
        seq_len: 输入序列长度
        pred_len: 预测步长

    Returns:
        (timeline_true, timeline_pred): 两个一维数组，按时间顺序对应
    """
    pred_sum = np.zeros(event_len, dtype=np.float64)
    pred_count = np.zeros(event_len, dtype=np.int32)
    true_values = np.full(event_len, np.nan, dtype=np.float64)

    n_samples = preds_event.shape[0]
    for sample_idx in range(n_samples):
        for step in range(pred_len):
            target_idx = sample_idx + seq_len + step
            if target_idx >= event_len:
                continue

            pred_sum[target_idx] += preds_event[sample_idx, step]
            pred_count[target_idx] += 1

            if np.isnan(true_values[target_idx]):
                true_values[target_idx] = labels_event[sample_idx, step]

    valid_mask = pred_count > 0
    timeline_pred = pred_sum[valid_mask] / pred_count[valid_mask]
    timeline_true = true_values[valid_mask]

    return timeline_true, timeline_pred


def model_test(model, device, args):
    """
    模型测试/预测函数
    
    Args:
        model: 模型实例
        device: 设备
        args: 参数配置
    """
    print("\n" + "="*50)
    print("开始测试...")
    print("="*50)

    # 仅保留时间点级输出前缀
    timeline_prefix = "timeline"
    
    # 加载数据和事件信息
    _, scaler, events_info = prepare_datasets(
        site_name=args.site_name,
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    test_events = events_info['test_events']
    
    # 加载scaler和模型
    scaler.load(os.path.join(args.save_path, 'scaler.pkl'))
    if hasattr(model, 'set_scaler_stats'):
        model.set_scaler_stats(scaler.phys_mean, scaler.phys_std)

    if is_non_torch_baseline(model):
        model.load_baseline(os.path.join(args.save_path, 'baseline_model.pkl'))
    else:
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    
    # 重新加载原始数据用于逐事件预测
    df_dsd, df_params = load_preprocessed_data(args.site_name, args.data_dir)
    conc, vel = extract_dsd_features(df_dsd)
    phys = df_params[['RainRate', 'Dm', 'LogNw', 'LWC', 'Z']].values
    rain_rate = df_params['RainRate'].values
    conc = apply_log_transform(conc)
    
    # 逐事件预测
    event_preds = []  # list，每个元素是一个事件的预测结果 (N_event, pred_len)
    event_labels = []  # list，每个元素是一个事件的真实标签 (N_event, pred_len)
    event_lengths = []  # list，每个元素是一个事件的原始长度
    timeline_preds = []  # list，每个元素是事件级时间点去重后的预测
    timeline_labels = []  # list，每个元素是事件级时间点去重后的真实值
    
    print(f"\n测试集包含 {len(test_events)} 个事件")
    
    for evt_idx, evt in enumerate(test_events):
        # 提取该事件的数据
        event_indices = list(range(evt['start_idx'], evt['end_idx'] + 1))
        event_conc = conc[event_indices]
        event_vel = vel[event_indices]
        event_phys = phys[event_indices]
        event_rain = rain_rate[event_indices]
        
        # 对该事件生成滑动窗口样本
        X_conc, X_vel, X_phys, Y = create_sliding_window_samples(
            event_conc, event_vel, event_phys, event_rain,
            args.seq_len, args.pred_len, args.stride
        )
        
        if X_conc is None or len(X_conc) == 0:
            print(f"  事件 {evt_idx+1}: 跳过（长度不足）")
            continue
        
        # 统一特征缩放
        X_conc, X_vel, X_phys = scaler.transform(X_conc, X_vel, X_phys)

        if is_non_torch_baseline(model):
            preds_event = model.predict_baseline(X_conc, X_vel, X_phys)
        else:
            # 转换为Tensor
            X_conc_tensor = torch.FloatTensor(X_conc).to(device)
            X_vel_tensor = torch.FloatTensor(X_vel).to(device)
            X_phys_tensor = torch.FloatTensor(X_phys).to(device)

            # 预测（所有模型统一接口）
            with torch.no_grad():
                predictions = model(X_conc_tensor, X_vel_tensor, X_phys_tensor)
                predictions = apply_output_constraint(
                    predictions,
                    enforce_non_negative=args.enforce_non_negative
                )
                preds_event = predictions.cpu().numpy()  # (N_event, pred_len)

        preds_event = apply_output_constraint(
            preds_event,
            enforce_non_negative=args.enforce_non_negative
        )
        
        event_preds.append(preds_event)
        event_labels.append(Y)  # (N_event, pred_len)
        event_lengths.append(len(event_rain))

        # 聚合为唯一时间点序列，避免整体指标重复计数
        t_true, t_pred = aggregate_event_predictions_to_timeline(
            preds_event=preds_event,
            labels_event=Y,
            event_len=len(event_rain),
            seq_len=args.seq_len,
            pred_len=args.pred_len
        )
        if len(t_true) > 0:
            timeline_labels.append(t_true)
            timeline_preds.append(t_pred)
        
        print(f"  事件 {evt_idx+1}: {len(preds_event)} 个样本")
    
    total_window_samples = sum(len(p) for p in event_preds)
    print(f"\n总共生成了 {total_window_samples} 个窗口预测样本（用于时间点级聚合）")
    
    # 合并所有事件的时间点级结果用于整体评估
    all_timeline_preds = np.concatenate(timeline_preds, axis=0) if timeline_preds else np.array([])
    all_timeline_labels = np.concatenate(timeline_labels, axis=0) if timeline_labels else np.array([])
    
    if len(all_timeline_preds) == 0:
        print("警告：无法生成时间点级评估结果！")
        return
    
    print(f"时间点级结果形状: {all_timeline_preds.shape}")
    
    # 检查预测值和标签的尺度（用于确认是否需要反归一化）
    print(f"\n时间点级预测值统计: min={np.min(all_timeline_preds):.4f}, max={np.max(all_timeline_preds):.4f}, mean={np.mean(all_timeline_preds):.4f}, std={np.std(all_timeline_preds):.4f}")
    print(f"时间点级标签统计:   min={np.min(all_timeline_labels):.4f}, max={np.max(all_timeline_labels):.4f}, mean={np.mean(all_timeline_labels):.4f}, std={np.std(all_timeline_labels):.4f}")
    print("注意：预测值和标签应该在原始尺度（mm/h）上，如果尺度不匹配，可能需要检查模型输出或添加反归一化")
    
    # 保存结果（仅保留时间点级）
    with open(os.path.join(args.save_path, f'{timeline_prefix}_predictions_by_event.pkl'), 'wb') as f:
        pickle.dump(timeline_preds, f)
    with open(os.path.join(args.save_path, f'{timeline_prefix}_labels_by_event.pkl'), 'wb') as f:
        pickle.dump(timeline_labels, f)
    
    # 保存合并后的时间点级结果
    np.save(os.path.join(args.save_path, f'{timeline_prefix}_predictions.npy'), all_timeline_preds)
    np.save(os.path.join(args.save_path, f'{timeline_prefix}_labels.npy'), all_timeline_labels)
    
    # 计算评价指标（唯一时间点级）
    print("\n整体评价指标（唯一时间点级）:")
    print("-"*50)
    overall_metrics = compute_all_metrics(all_timeline_labels, all_timeline_preds, threshold=args.threshold)
    print_metrics(overall_metrics, prefix="Overall(timeline) ")
    
    # 时间点级逐步评价
    print("\n时间点级逐步评价指标:")
    print("-"*50)
    step_metrics = []
    for step in range(args.pred_len):
        step_labels_all = []
        step_preds_all = []
        for labels_evt, preds_evt, event_len in zip(event_labels, event_preds, event_lengths):
            t_true, t_pred = aggregate_event_predictions_for_step(
                preds_event=preds_evt,
                labels_event=labels_evt,
                event_len=event_len,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                stride=args.stride,
                step=step
            )
            if len(t_true) == 0:
                continue
            step_labels_all.extend(t_true)
            step_preds_all.extend(t_pred)

        if len(step_labels_all) == 0:
            print(f"\nStep {step+1} (T+{step+1}) 无有效时间点，已跳过")
            continue

        metrics = compute_all_metrics(np.array(step_labels_all), np.array(step_preds_all), threshold=args.threshold)
        step_metrics.append((step + 1, metrics))
        print(f"\nStep {step+1} (T+{step+1}):")
        print_metrics(metrics, prefix="  ")
    
    # 保存指标到文件
    write_timeline_metrics_report(
        save_path=args.save_path,
        model_name=args.model_name,
        site_name=args.site_name,
        overall_metrics=overall_metrics,
        step_metrics=step_metrics
    )
    
    # 绘制时间点级预测结果
    plot_timeline_aggregated_predictions(
        timeline_labels,
        timeline_preds,
        args.save_path,
        threshold=args.threshold
    )
    plot_timeline_stepwise_predictions(
        event_labels=event_labels,
        event_preds=event_preds,
        event_lengths=event_lengths,
        save_path=args.save_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        threshold=args.threshold
    )
    plot_timeline_stepwise_panel(
        event_labels=event_labels,
        event_preds=event_preds,
        event_lengths=event_lengths,
        timeline_labels=timeline_labels,
        timeline_preds=timeline_preds,
        save_path=args.save_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        threshold=args.threshold
    )
    
    print("\n" + "="*50)
    print("测试完成！")
    print(f"结果已保存到: {args.save_path}")
    print("="*50)
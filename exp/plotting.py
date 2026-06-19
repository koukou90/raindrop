import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.metrics import compute_all_metrics

# 预测对比图统一字号配置（论文可读性优先）
PLOT_LABEL_FONTSIZE = 14
PLOT_TICK_FONTSIZE = 12
PLOT_LEGEND_FONTSIZE = 13
PLOT_TITLE_FONTSIZE = 15
PLOT_METRICS_FONTSIZE = 11

# 2x3 面板图单子图字号（面板内略小于单图）
PANEL_LABEL_FONTSIZE = 12
PANEL_TICK_FONTSIZE = 10.5
PANEL_LEGEND_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 13.5
PANEL_METRICS_FONTSIZE = 10

# 论文图统一视觉样式
TRUE_LINE_COLOR = '#d62728'
PRED_LINE_COLOR = '#1f77b4'
BOUNDARY_COLOR = '#9e9e9e'

PAPER_PLOT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.color': '#d9d9d9',
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.7,
}


def aggregate_event_predictions_for_step(
    preds_event,
    labels_event,
    event_len,
    seq_len,
    pred_len,
    stride,
    step
):
    """
    将单个预测步长在事件内映射到唯一时间点序列。
    """
    pred_sum = np.zeros(event_len, dtype=np.float64)
    pred_count = np.zeros(event_len, dtype=np.int32)
    true_values = np.full(event_len, np.nan, dtype=np.float64)

    start_indices = list(range(0, event_len - seq_len - pred_len + 1, stride))
    n_samples = min(len(start_indices), preds_event.shape[0])

    for sample_idx in range(n_samples):
        target_idx = start_indices[sample_idx] + seq_len + step
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


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['learning_rate'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"训练曲线已保存到: {os.path.join(save_path, 'training_curves.png')}")


def plot_timeline_aggregated_predictions(timeline_labels, timeline_preds, save_path, threshold=20.0):
    """
    绘制时间点级聚合曲线图（与 Overall(timeline) 评估口径一致）。
    """
    if len(timeline_labels) == 0 or len(timeline_preds) == 0:
        print("时间点级聚合曲线图跳过：没有可用数据")
        return

    with plt.rc_context(PAPER_PLOT_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        concatenated_labels = []
        concatenated_preds = []
        event_boundaries = [0]

        for labels_evt, preds_evt in zip(timeline_labels, timeline_preds):
            concatenated_labels.extend(labels_evt)
            concatenated_preds.extend(preds_evt)
            event_boundaries.append(len(concatenated_labels))

        concatenated_labels = np.array(concatenated_labels)
        concatenated_preds = np.array(concatenated_preds)

        x_indices = np.arange(len(concatenated_labels))
        ax.plot(x_indices, concatenated_labels, color=TRUE_LINE_COLOR, label='True', linewidth=1.1, alpha=0.9)
        ax.plot(x_indices, concatenated_preds, color=PRED_LINE_COLOR, label='Pred', linewidth=1.1, alpha=0.9)

        for boundary in event_boundaries[1:-1]:
            ax.axvline(x=boundary, color=BOUNDARY_COLOR, linestyle='--', linewidth=0.7, alpha=0.35)

        overall_metrics = compute_all_metrics(concatenated_labels, concatenated_preds, threshold=threshold)
        peak_mae_str = f"{overall_metrics['Peak_MAE']:.3f}" if not np.isnan(overall_metrics['Peak_MAE']) else "N/A"
        metrics_text = (
            f"RMSE={overall_metrics['RMSE']:.3f} | "
            f"MAE={overall_metrics['MAE']:.3f} | "
            f"CC={overall_metrics['CC']:.3f} | "
            f"Peak-MAE={peak_mae_str}"
        )
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=PLOT_METRICS_FONTSIZE,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bdbdbd', alpha=0.9)
        )

        ax.set_xlabel('Timeline Index (Concatenated by Events)', fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_ylabel('Rain Rate (mm/h)', fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_title(
            f'All Events - Timeline Aggregated Prediction (Total Points: {len(concatenated_labels)}, Events: {len(timeline_labels)})',
            fontsize=PLOT_TITLE_FONTSIZE,
            fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=PLOT_LEGEND_FONTSIZE, frameon=False)
        ax.tick_params(axis='both', labelsize=PLOT_TICK_FONTSIZE, direction='out', length=3, width=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'timeline_predictions_aggregated.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    print("时间点级聚合预测对比图已保存（timeline 前缀）")


def collect_timeline_stepwise_series(
    event_labels,
    event_preds,
    event_lengths,
    seq_len,
    pred_len,
    stride,
    threshold=20.0,
    selected_steps=None
):
    """
    收集时间点级分步序列与指标，供单图/面板图复用。
    """
    if len(event_labels) == 0 or len(event_preds) == 0:
        return []

    if selected_steps is None:
        selected_steps = list(range(pred_len))

    series = []
    for step in selected_steps:
        if step < 0 or step >= pred_len:
            continue

        concatenated_labels = []
        concatenated_preds = []
        event_boundaries = [0]

        for labels_evt, preds_evt, event_len in zip(event_labels, event_preds, event_lengths):
            t_true, t_pred = aggregate_event_predictions_for_step(
                preds_event=preds_evt,
                labels_event=labels_evt,
                event_len=event_len,
                seq_len=seq_len,
                pred_len=pred_len,
                stride=stride,
                step=step
            )

            if len(t_true) == 0:
                continue

            concatenated_labels.extend(t_true)
            concatenated_preds.extend(t_pred)
            event_boundaries.append(len(concatenated_labels))

        if len(concatenated_labels) == 0:
            continue

        y_true = np.array(concatenated_labels)
        y_pred = np.array(concatenated_preds)
        metrics = compute_all_metrics(y_true, y_pred, threshold=threshold)
        series.append({
            'step': step + 1,
            'y_true': y_true,
            'y_pred': y_pred,
            'event_boundaries': event_boundaries,
            'metrics': metrics
        })

    return series


def plot_timeline_stepwise_predictions(
    event_labels,
    event_preds,
    event_lengths,
    save_path,
    seq_len,
    pred_len,
    stride,
    threshold=20.0,
    selected_steps=None
):
    """
    绘制时间点级分步图（T+1...T+pred_len），每个步长一张图。
    """
    series = collect_timeline_stepwise_series(
        event_labels=event_labels,
        event_preds=event_preds,
        event_lengths=event_lengths,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        threshold=threshold,
        selected_steps=selected_steps
    )

    if len(series) == 0:
        print("时间点级分步图跳过：没有可用数据")
        return

    with plt.rc_context(PAPER_PLOT_STYLE):
        for item in series:
            step = item['step']
            y_true = item['y_true']
            y_pred = item['y_pred']
            event_boundaries = item['event_boundaries']
            step_metrics = item['metrics']

            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            x_indices = np.arange(len(y_true))
            ax.plot(x_indices, y_true, color=TRUE_LINE_COLOR, label='True', linewidth=1.1, alpha=0.9)
            ax.plot(x_indices, y_pred, color=PRED_LINE_COLOR, label='Pred', linewidth=1.1, alpha=0.9)

            for boundary in event_boundaries[1:-1]:
                ax.axvline(x=boundary, color=BOUNDARY_COLOR, linestyle='--', linewidth=0.7, alpha=0.35)

            peak_mae_str = f"{step_metrics['Peak_MAE']:.3f}" if not np.isnan(step_metrics['Peak_MAE']) else "N/A"
            metrics_text = (
                f"RMSE={step_metrics['RMSE']:.3f} | "
                f"MAE={step_metrics['MAE']:.3f} | "
                f"CC={step_metrics['CC']:.3f} | "
                f"Peak-MAE={peak_mae_str}"
            )
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=PLOT_METRICS_FONTSIZE,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bdbdbd', alpha=0.9)
            )

            ax.set_xlabel('Timeline Index (Concatenated by Events)', fontsize=PLOT_LABEL_FONTSIZE)
            ax.set_ylabel('Rain Rate (mm/h)', fontsize=PLOT_LABEL_FONTSIZE)
            ax.set_title(
                f'All Events - Timeline Stepwise Prediction at T+{step} (Total Points: {len(y_true)})',
                fontsize=PLOT_TITLE_FONTSIZE,
                fontweight='bold'
            )
            ax.legend(loc='upper right', fontsize=PLOT_LEGEND_FONTSIZE, frameon=False)
            ax.tick_params(axis='both', labelsize=PLOT_TICK_FONTSIZE, direction='out', length=3, width=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_path, f'timeline_predictions_T{step}_stepwise.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)
            print(f"  T+{step} 时间点级分步图已保存（timeline 前缀）")

    print(f"时间点级分步图已保存到: {save_path}")


def plot_timeline_stepwise_panel(
    event_labels,
    event_preds,
    event_lengths,
    timeline_labels,
    timeline_preds,
    save_path,
    seq_len,
    pred_len,
    stride,
    threshold=20.0,
    selected_steps=None
):
    """
    生成论文风格 2x3 面板图（5个步长 + 1个聚合面板）。
    """
    series = collect_timeline_stepwise_series(
        event_labels=event_labels,
        event_preds=event_preds,
        event_lengths=event_lengths,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        threshold=threshold,
        selected_steps=selected_steps
    )
    if len(series) == 0:
        print("时间点级 2x3 面板图跳过：没有可用数据")
        return

    agg_true = None
    agg_pred = None
    agg_boundaries = None
    agg_metrics = None
    if len(timeline_labels) > 0 and len(timeline_preds) > 0:
        agg_true = []
        agg_pred = []
        agg_boundaries = [0]
        for labels_evt, preds_evt in zip(timeline_labels, timeline_preds):
            agg_true.extend(labels_evt)
            agg_pred.extend(preds_evt)
            agg_boundaries.append(len(agg_true))
        agg_true = np.array(agg_true)
        agg_pred = np.array(agg_pred)
        agg_metrics = compute_all_metrics(agg_true, agg_pred, threshold=threshold)

    value_arrays = [np.concatenate([item['y_true'], item['y_pred']]) for item in series]
    if agg_true is not None and agg_pred is not None:
        value_arrays.append(np.concatenate([agg_true, agg_pred]))
    all_values = np.concatenate(value_arrays)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_margin = max(0.1, (y_max - y_min) * 0.05)

    with plt.rc_context(PAPER_PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes = axes.flatten()
        legend_handles = None

        for idx, item in enumerate(series[:5]):
            ax = axes[idx]
            step = item['step']
            y_true = item['y_true']
            y_pred = item['y_pred']
            event_boundaries = item['event_boundaries']
            metrics = item['metrics']
            x_indices = np.arange(len(y_true))

            line_true, = ax.plot(x_indices, y_true, color=TRUE_LINE_COLOR, linewidth=1.1, alpha=0.9)
            line_pred, = ax.plot(x_indices, y_pred, color=PRED_LINE_COLOR, linewidth=1.1, alpha=0.9)
            if legend_handles is None:
                legend_handles = [line_true, line_pred]

            for boundary in event_boundaries[1:-1]:
                ax.axvline(x=boundary, color=BOUNDARY_COLOR, linestyle='--', linewidth=0.7, alpha=0.35)

            panel_tag = chr(ord('a') + idx)
            ax.set_title(f'({panel_tag}) T+{step}', fontsize=PANEL_TITLE_FONTSIZE, fontweight='bold')
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.tick_params(labelsize=PANEL_TICK_FONTSIZE, direction='out', length=3, width=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if idx % 3 == 0:
                ax.set_ylabel('Rain Rate (mm/h)', fontsize=PANEL_LABEL_FONTSIZE)
            if idx >= 3:
                ax.set_xlabel('Timeline Index', fontsize=PANEL_LABEL_FONTSIZE)

            ax.text(
                0.02,
                0.98,
                f"RMSE={metrics['RMSE']:.3f}\nMAE={metrics['MAE']:.3f}\nCC={metrics['CC']:.3f}",
                transform=ax.transAxes,
                fontsize=PANEL_METRICS_FONTSIZE,
                va='top',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bdbdbd', alpha=0.9)
            )

        agg_ax = axes[5]
        if agg_true is not None and agg_pred is not None:
            agg_x = np.arange(len(agg_true))
            line_true, = agg_ax.plot(agg_x, agg_true, color=TRUE_LINE_COLOR, linewidth=1.1, alpha=0.9)
            line_pred, = agg_ax.plot(agg_x, agg_pred, color=PRED_LINE_COLOR, linewidth=1.1, alpha=0.9)
            if legend_handles is None:
                legend_handles = [line_true, line_pred]

            for boundary in agg_boundaries[1:-1]:
                agg_ax.axvline(x=boundary, color=BOUNDARY_COLOR, linestyle='--', linewidth=0.7, alpha=0.35)

            agg_ax.set_title('(f) Aggregated', fontsize=PANEL_TITLE_FONTSIZE, fontweight='bold')
            agg_ax.set_ylim(y_min - y_margin, y_max + y_margin)
            agg_ax.tick_params(labelsize=PANEL_TICK_FONTSIZE, direction='out', length=3, width=0.8)
            agg_ax.spines['top'].set_visible(False)
            agg_ax.spines['right'].set_visible(False)
            agg_ax.set_xlabel('Timeline Index', fontsize=PANEL_LABEL_FONTSIZE)

            agg_ax.text(
                0.02,
                0.98,
                f"RMSE={agg_metrics['RMSE']:.3f}\nMAE={agg_metrics['MAE']:.3f}\nCC={agg_metrics['CC']:.3f}",
                transform=agg_ax.transAxes,
                fontsize=PANEL_METRICS_FONTSIZE,
                va='top',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bdbdbd', alpha=0.9)
            )
        else:
            agg_ax.axis('off')
            agg_ax.text(0.5, 0.5, '(f) Aggregated\nNo data', ha='center', va='center', fontsize=11)

        if legend_handles is not None:
            fig.legend(
                legend_handles,
                ['True', 'Pred'],
                loc='upper center',
                ncol=2,
                frameon=False,
                fontsize=PANEL_LEGEND_FONTSIZE,
                bbox_to_anchor=(0.5, 0.995)
            )

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        panel_png = os.path.join(save_path, 'timeline_predictions_stepwise_panel_2x3.png')
        fig.savefig(panel_png, dpi=600, bbox_inches='tight')
        plt.close(fig)

    print("时间点级分步 2x3 面板图已保存（论文风格，PNG）")

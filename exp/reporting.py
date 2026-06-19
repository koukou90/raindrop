import os
import numpy as np


def write_timeline_metrics_report(save_path, model_name, site_name, overall_metrics, step_metrics):
    """
    将时间点级整体与分步评价指标写入测试报告文件。
    """
    report_path = os.path.join(save_path, 'test_metrics.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"站点: {site_name}\n")
        f.write("=" * 90 + "\n\n")

        f.write("整体评价指标（时间点级，论文主口径）:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'RMSE':<10} {'MAE':<10} {'MSE':<10} {'CC':<10} {'R²':<10} {'MAPE':<10} {'Bias':<10} {'NSE':<10} {'Peak-MAE':<12}\n")
        f.write("-" * 120 + "\n")

        mape_str = f"{overall_metrics['MAPE']:.4f}" if not np.isnan(overall_metrics['MAPE']) else "N/A"
        nse_str = f"{overall_metrics['NSE']:.4f}" if not np.isnan(overall_metrics['NSE']) else "N/A"
        peak_mae_str = f"{overall_metrics['Peak_MAE']:.4f}" if not np.isnan(overall_metrics['Peak_MAE']) else "N/A"
        f.write(
            f"{overall_metrics['RMSE']:<10.4f} {overall_metrics['MAE']:<10.4f} {overall_metrics['MSE']:<10.4f} "
            f"{overall_metrics['CC']:<10.4f} {overall_metrics['R2']:<10.4f} {mape_str:<10} "
            f"{overall_metrics['Bias']:<10.4f} {nse_str:<10} {peak_mae_str:<12}\n"
        )

        f.write("\n时间点级逐步评价指标（表格形式，便于对比不同步长）:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Step':<8} {'RMSE':<10} {'MAE':<10} {'MSE':<10} {'CC':<10} {'R²':<10} {'MAPE':<10} {'Bias':<10} {'NSE':<10} {'Peak-MAE':<12}\n")
        f.write("-" * 120 + "\n")

        for step, metrics in step_metrics:
            mape_str = f"{metrics['MAPE']:.4f}" if not np.isnan(metrics['MAPE']) else "N/A"
            nse_str = f"{metrics['NSE']:.4f}" if not np.isnan(metrics['NSE']) else "N/A"
            peak_mae_str = f"{metrics['Peak_MAE']:.4f}" if not np.isnan(metrics['Peak_MAE']) else "N/A"
            f.write(
                f"T+{step:<6} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['MSE']:<10.4f} "
                f"{metrics['CC']:<10.4f} {metrics['R2']:<10.4f} {mape_str:<10} "
                f"{metrics['Bias']:<10.4f} {nse_str:<10} {peak_mae_str:<12}\n"
            )

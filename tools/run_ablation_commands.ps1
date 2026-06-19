# 三站点 A 类结构消融一键运行命令清单
# 用法：在项目根目录依次执行（或运行 python tools/run_all_ablations.py）

$SITES = @('W2127_Haichaoba', 'W2128_Haichaoyinsi', 'W2129_buligou')

# A类：结构消融（5个）
# 说明：A类会自动套用 self_distill 训练策略
foreach ($site in $SITES) {
  python run.py --model_name AblateV3NoPersistence --site_name $site
  python run.py --model_name AblateV3NoMixGate --site_name $site
  python run.py --model_name AblateV3NoBinAware --site_name $site
  python run.py --model_name AblateV3NoAux --site_name $site
  python run.py --model_name AblateV3NoStreamGate --site_name $site
}

# 回填各站点消融 xlsx
python tools/update_ablation_metrics_xlsx.py --site_name all

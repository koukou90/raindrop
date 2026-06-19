import pickle
import numpy as np


class LightGBMBaseline:
    """LightGBM 多步回归基线（每个预测步训练一个回归器）。"""

    # LightGBM 默认训练配置（统一内聚在模型文件中）
    DEFAULT_CONFIG = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'early_stopping_rounds': 30,
    }
    CONFIG_KEYS = tuple(DEFAULT_CONFIG.keys())

    def __init__(
        self,
        args=None,
        pred_len=5,
        config=None
    ):
        if args is not None:
            pred_len = getattr(args, 'pred_len', pred_len)
            self.random_state = getattr(args, 'seed', 2025)
        else:
            self.random_state = 2025

        cfg = dict(self.DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)
        self.config = cfg

        self.pred_len = pred_len
        for key in self.CONFIG_KEYS:
            setattr(self, key, cfg[key])
        self.models = []
        self.is_fitted = False

        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBM 未安装。请先执行: pip install lightgbm"
            ) from exc

    @staticmethod
    def _build_features(conc, vel, phys):
        x_conc = conc.reshape(conc.shape[0], -1)
        x_vel = vel.reshape(vel.shape[0], -1)
        x_phys = phys.reshape(phys.shape[0], -1)
        return np.concatenate([x_conc, x_vel, x_phys], axis=1)

    def _dataset_to_xy(self, dataset):
        x = self._build_features(
            dataset.conc_data.numpy(),
            dataset.vel_data.numpy(),
            dataset.phys_data.numpy()
        )
        y = dataset.labels.numpy()
        return x, y

    def fit_baseline(self, train_dataset, val_dataset=None):
        x_train, y_train = self._dataset_to_xy(train_dataset)

        if val_dataset is not None:
            x_val, y_val = self._dataset_to_xy(val_dataset)
        else:
            x_val = None
            y_val = None

        models = []
        for step in range(self.pred_len):
            reg = self.lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )

            fit_kwargs = {}
            if x_val is not None:
                fit_kwargs['eval_set'] = [(x_val, y_val[:, step])]
                fit_kwargs['eval_metric'] = 'l2'
                fit_kwargs['callbacks'] = [
                    self.lgb.early_stopping(
                        stopping_rounds=self.early_stopping_rounds,
                        verbose=False
                    )
                ]

            reg.fit(x_train, y_train[:, step], **fit_kwargs)
            models.append(reg)

        self.models = models
        self.is_fitted = True

    def predict_baseline(self, conc, vel, phys):
        if not self.is_fitted:
            raise RuntimeError("LightGBM baseline 尚未训练，请先执行 fit_baseline。")

        x = self._build_features(conc, vel, phys)
        preds = [m.predict(x).reshape(-1, 1) for m in self.models]
        return np.concatenate(preds, axis=1)

    def save_baseline(self, filepath):
        payload = {
            'pred_len': self.pred_len,
            'config': self.config,
            'random_state': self.random_state,
            'models': self.models,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)

    def load_baseline(self, filepath):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)

        self.pred_len = payload['pred_len']
        # 兼容旧版本：旧模型文件没有 config 字段
        if 'config' in payload:
            self.config = payload['config']
        else:
            self.config = {
                'n_estimators': payload['n_estimators'],
                'learning_rate': payload['learning_rate'],
                'num_leaves': payload['num_leaves'],
                'subsample': payload['subsample'],
                'colsample_bytree': payload['colsample_bytree'],
                'early_stopping_rounds': payload['early_stopping_rounds'],
            }
        for key in self.CONFIG_KEYS:
            setattr(self, key, self.config[key])
        self.random_state = payload['random_state']
        self.models = payload['models']
        self.is_fitted = payload['is_fitted']

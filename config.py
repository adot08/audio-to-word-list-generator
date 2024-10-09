import yaml
import os
from typing import Any

class Config:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, 'config.yaml'), 'r') as config_file:
            self._config = yaml.safe_load(config_file)

        # 设置 NLTK 数据路径
        self._config['nltk_data_path'] = os.path.join(base_dir, self._config['nltk_data_path'])

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            # 检查是否有环境变量覆盖
            env_value = os.getenv(name.upper())
            return env_value if env_value is not None else self._config[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")

# 创建一个全局配置实例
config = Config()
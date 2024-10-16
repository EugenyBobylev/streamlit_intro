import os
from pathlib import Path


from dotenv import load_dotenv


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    def __init__(self):
        script_path = os.path.abspath(__file__)
        env_path = Path(script_path).parent / '.env'

        load_dotenv(env_path)
        self.app_dir: str = str(Path(script_path).parent)

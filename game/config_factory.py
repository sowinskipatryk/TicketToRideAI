from game.enums import GameVersion
from game.game_config import BaseConfig, USAConfig, EuropeConfig, NordicConfig


class ConfigFactory:
    @staticmethod
    def get_config(version: str) -> BaseConfig:
        try:
            game_version = GameVersion(version)
        except ValueError:
            raise ValueError("Invalid game version. Choose from: USA, Europe, Nordic")

        config_map = {
            GameVersion.USA: USAConfig,
            GameVersion.EUROPE: EuropeConfig,
            GameVersion.NORDIC: NordicConfig,
            }

        config_class = config_map.get(game_version)
        return config_class()

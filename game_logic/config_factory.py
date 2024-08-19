from game_logic.enums.game_versions import GameVersion
from game_logic.game_config import BaseGameConfig, GameConfigUSA, GameConfigEurope, GameConfigNordic


class ConfigFactory:
    @staticmethod
    def get_config(version: str) -> BaseGameConfig:
        try:
            game_version = GameVersion(version)
        except ValueError:
            raise ValueError("Invalid game version. Choose from: USA, Europe, Nordic")

        config_map = {
            GameVersion.USA: GameConfigUSA,
            GameVersion.EUROPE: GameConfigEurope,
            GameVersion.NORDIC: GameConfigNordic,
            }

        config_class = config_map.get(game_version)
        return config_class()

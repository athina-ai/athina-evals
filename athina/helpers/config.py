import yaml

CONFIG_FILE_NAME = "athina_config.yml"


class ConfigHelper:
    @staticmethod
    def load_config():
        try:
            with open(CONFIG_FILE_NAME, "r") as file:
                config = yaml.safe_load(file)

            if config is None:
                config = {}
            return config
        except:
            return {}

    @staticmethod
    def save_config(config_data):
        with open(CONFIG_FILE_NAME, "w") as file:
            yaml.dump(config_data, file)

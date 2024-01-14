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
    def load_config_field(field: str):
        try:
            config = ConfigHelper.load_config()
            return config[field]
        except Exception as e:
            return None

    @staticmethod
    def load_openai_api_key():
        return ConfigHelper.load_config_field("openai_api_key")

    @staticmethod
    def load_athina_api_key():
        return ConfigHelper.load_config_field("athina_api_key")

    @staticmethod
    def load_llm_engine():
        return ConfigHelper.load_config_field("llm_engine")

    @staticmethod
    def save_config(config_data):
        with open(CONFIG_FILE_NAME, "w") as file:
            yaml.dump(config_data, file)

    @staticmethod
    def is_set():
        try:
            with open(CONFIG_FILE_NAME, "r") as file:
                config = yaml.safe_load(file)

            if config is None or config == {}:
                return False
            else:
                return True
        except:
            return False

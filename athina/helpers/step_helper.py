import json

class StepHelper:
    
    @staticmethod
    def prepare_input_data(data):
        return {
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in data.items()
        }
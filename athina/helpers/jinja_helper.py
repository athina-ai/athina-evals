from jinja2 import Undefined

class PreserveUndefined(Undefined):
    def __str__(self):
        return f'{{ {self._undefined_name} }}'
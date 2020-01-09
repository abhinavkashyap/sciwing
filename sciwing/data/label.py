class Label:
    def __init__(self, label_str: str):
        self.label_str = label_str

    @property
    def label_str(self):
        return self._label_str

    @label_str.setter
    def label_str(self, value):
        self._label_str = value

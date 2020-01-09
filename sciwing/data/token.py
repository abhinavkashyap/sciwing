class Token:
    def __init__(self, text: str):
        self.text = text

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def len(self):
        return len(self.text)

class ClassInNurseryError(KeyError):
    pass


class TOMLConfigurationError(Exception):
    def __init__(self, message: str):
        super(TOMLConfigurationError, self).__init__()
        self.message = message

    def __repr__(self):
        return self.message

    def __str__(self):
        return repr(self)


class DatasetPresentError(Exception):
    def __init__(self, message: str):
        super(DatasetPresentError, self).__init__()
        self.message = message

    def __repr__(self):
        return self.message

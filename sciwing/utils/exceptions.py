class ClassInNurseryError(KeyError):
    """The ClassNursery cannot have two classes of the same name. This error is raised when
    that happens
    """

    pass


class TOMLConfigurationError(Exception):
    """ This error is raised for illegal configuration of TOML

    """

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

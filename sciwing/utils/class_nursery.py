from sciwing.utils.exceptions import ClassInNurseryError
import torch


class ClassNursery(object):
    """ClassNursery is the place where all the classes in SciWING are nursed

    SciWING needs to get handle on the different classes that are being used.
    This is further useful for example, when we have to instantiate appropriate
    classes when the experiments are run from the TOML file

    This uses a python 36 feature called __init_subclass__ that simplifies
    class creation. Whenever ClassNursery is mentioned as the parent class of a class,
    then init subclass is called. In SciWING we use it as a plugin registry where the
    mapping between the different class and their module is stored.

    """

    class_nursery = {"Adam": torch.optim.__name__, "SGD": torch.optim.__name__}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.class_nursery.get(cls.__name__) is None:
            cls.class_nursery[cls.__name__] = cls.__module__
        else:
            raise ClassInNurseryError(
                f"Class {cls.__name__} present in Nursery."
                f"Please chose another class name"
            )

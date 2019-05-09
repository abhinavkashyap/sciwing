from typing import List
from stop_words import get_stop_words


class InstancePreprocessing:
    """
    This class implements some common pre-processing that may be applied on
    instances which are List[str]. For example, you can remove stop words, convert the word
    into lower case and others. Most of the methods here accept an instance and return an instance
    """
    def __init__(self):
        self.stop_words = get_stop_words("en")

    def lowercase(self, instance:List[str]) -> List[str]:
        lowercased_instance = []
        for token in instance:
            lowercased_instance.append(token.lower())

        return lowercased_instance

    def remove_stop_words(self, instance:List[str]) -> List[str]:
        """
        Remove stop words if they are present
        We will use stop-words package from pip
        https://github.com/Alir3z4/python-stop-words
        :param instance:
        :return:
        """
        clean_instance = filter(lambda token: token not in self.stop_words, instance)
        clean_instance = list(clean_instance)
        return clean_instance

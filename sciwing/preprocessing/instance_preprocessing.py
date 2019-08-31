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

    @staticmethod
    def lowercase(instance: List[str]) -> List[str]:
        lowercased_instance = []
        for token in instance:
            lowercased_instance.append(token.lower())

        return lowercased_instance

    def remove_stop_words(self, instance: List[str]) -> List[str]:
        """
        Remove stop words if they are present
        We will use stop-words package from pip
        https://github.com/Alir3z4/python-stop-words

        Parameters
        --------------
        instance : List[str]
            The list of tokens

        Returns
        ---------
        List[str]
            The instance with stop words removed

        """
        clean_instance = filter(lambda token: token not in self.stop_words, instance)
        clean_instance = list(clean_instance)
        return clean_instance

    @staticmethod
    def indicate_capitalization(instance: List[str]) -> List[str]:
        """ Indicates whether every word is all small, all caps or captialized

        Parameters
        ----------
        instance : List[str]
            A list of tokens
        Returns
        -------
        List[str]
            Strings indicating capitalization
        """
        processed_instance = []
        for word in instance:
            processed_word = "[OTHER]"
            if word.istitle() and word.isalpha():
                processed_word = "[CAPITALIZED]"
            if word.islower() and word.isalpha():
                processed_word = "[LOWER]"
            if word.isupper() and word.isalpha():
                processed_word = "[UPPER]"
            if word.isnumeric():
                processed_word = "[NUMERIC]"
            if word.isalnum() and not processed_word != "[OTHER]":
                processed_word = "[ALNUM]"

            processed_instance.append(processed_word)

        assert len(instance) == len(
            processed_instance
        ), f"Length instance {len(instance)}, Lenght of Processed Instance {len(processed_instance)}"
        return processed_instance

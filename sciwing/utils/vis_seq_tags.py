from typing import List, Dict, Any
import itertools
import colorful
from sciwing.utils.common import pairwise


class VisTagging:
    def __init__(
        self,
        colors: List[str] = None,
        colors_palette: str = None,
        tags: List[str] = None,
    ):

        """ Visualize Sequence Tagging

        Parameters
        ----------
        colors: List[str]
            The set of colors that will be used for tagging
        colors_palette: str
            The color palette that should be used. We recommend
            For more information on color palettes  you can refer to the documentation of the python package `colorful`
        tags: List[str]
            The set of all labels that can be labelled
            If this is not given, then the tags will be infered using the labels during tagging

        """
        if colors is None:
            colors = [
                "yellow",
                "orange",
                "red",
                "magenta",
                "violet",
                "blue",
                "cyan",
                "green",
            ]
        if colors_palette is None:
            colors_palette = "solarized"

        self.colors = colors
        self.colors_palette = colors_palette
        self.colors_iter = itertools.cycle(self.colors)
        self.tags = tags
        colorful.use_style(self.colors_palette)

    def _get_next_color(self):
        """ Returns the next colors in the palette. Recycle the colors

        Returns
        -------

        """
        return next(self.colors_iter)

    def visualize_tags_from_json(
        self, json_annotation: Dict[str, Any], show_only_entities: List[str] = None
    ):
        """ Visualize the tags from json.

        Parameters
        ----------
        json_annotation: str
            You can send a json that has the following format
            {'text': str,
            'tags': [{'start':int, 'end':str, 'tag': str}]
            }
        show_only_entities: List[str]
            You can filter to show only these entities.

        """
        text = json_annotation.get("text", None)
        tags = json_annotation.get("tags", None)

        if not text or not tags:
            raise ValueError(
                "The json string should have the format "
                "{text:str, tags: [{'start': int, 'end': int, 'tag': str}]}"
            )

        if not self.tags:
            all_tags = list(set(annotation["tag"] for annotation in tags))
        else:
            all_tags = self.tags

        len_tags = len(all_tags)
        tag_colors = {
            all_tags[idx]: f"{{c.on_{self._get_next_color()}}}"
            for idx in range(len_tags)
        }

        valid_tags = {}
        for tag in tags:
            valid_tags[(tag["start"], tag["end"])] = tag["tag"]

        # assuming that the tags are given in order
        start_ends = []
        for tag in tags:
            start = tag["start"]
            end = tag["end"]
            start_ends.extend([start, end])

        if 0 not in start_ends:
            start_ends.insert(0, 0)
        if len(text) not in start_ends:
            start_ends.append(len(text))

        formatted_strings = []
        for start, end in pairwise(start_ends):
            if valid_tags.get((start, end), None) is not None:
                tag = valid_tags[(start, end)]
                if show_only_entities and tag not in show_only_entities:
                    continue
                tag_color = tag_colors[tag]
                formatted_string = f"{tag_color} {text[start:end]} {{c.bold}}{tag} {colorful.close_bg_color}"
                formatted_string = colorful.format(formatted_string)
            else:
                formatted_string = text[start:end]

            formatted_strings.append(formatted_string)
        tagged_string = " ".join(formatted_strings)
        print(tagged_string)

    def visualize_tokens(self, text: List[str], labels: List[str]) -> str:
        """ Visualizes sequential tagged data where the string is represented as a set of words
        and every word has a corresponding label. This can be extended to having different
        tagging schemes at a later point in time

        Parameters
        ----------
        text: List[str]
        String to be tagged represented as a list of strings

        labels: List[str]
        The labels corresponding to each word in the string

        Returns
        -------
        None
        """
        if len(text) != len(labels):
            raise ValueError(
                f"string and labels should of same length. String you passed has len {len(text)} "
                f"and labels you passed has len {len(labels)}"
            )

        if not self.tags:
            unique_labels = list(set(labels))
        else:
            unique_labels = self.tags
        len_labels = len(unique_labels)

        tag_colors = {
            unique_labels[idx]: f"{{c.on_{self._get_next_color()}}}"
            for idx in range(len_labels)
        }

        stylized_words = []

        for idx, word in enumerate(text):
            tag = labels[idx]
            tag_color = tag_colors[tag]

            formatted_string = (
                f"{colorful.reset}{tag_color}{word.strip()} "
                f"{{c.bold}}{tag.strip().upper()}{colorful.close_bg_color}"
            )
            formatted_string = colorful.format(formatted_string)
            stylized_words.append(formatted_string)

        tagged_string = " ".join(stylized_words)
        return tagged_string

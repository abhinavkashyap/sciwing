import torch
from typing import Union


def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    From ``allennlp.nn.util``
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_to_device(obj, cuda_device: torch.device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    From ``allenlp.nn.util``
    """
    # pylint: disable=too-many-return-statements
    # pargma: no cover
    # not tested relying on allennlp
    if cuda_device.type == "cpu" or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*[move_to_device(item, cuda_device) for item in obj])
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, cuda_device) for item in obj])
    else:
        return obj


def get_mask(batch_size: int, max_size: int, lengths: torch.LongTensor):
    """ Returns mask given the lengths tensor. A convenience method

    Given a lengths tensor as in

    .. code-block:: python

        >> torch.LongTensor([3, 1, 2])

    which often indicates the original length of the tensor
    without padding, `get_mask()` returns a tensor with 1 positions
    where there is no padding and 0 where there is padding

    Parameters
    ----------
    batch_size : int
        Batch size of the tensors
    max_size : int
        Maximum size or often Maximum number of time steps
    lengths : torch.LongTensor
        The original length of the tensors in the batch without padding

    Returns
    -------
    torch.LongTensor
        Mask having 1 where there are no paddings and 0 where there are paddings
    """
    assert batch_size == lengths.size(0)
    mask = []

    for length in lengths:
        zero_row = torch.zeros(max_size, dtype=torch.long)
        zero_row[: length.item()] = 1
        mask.append(zero_row.unsqueeze(0))

    mask = torch.cat(mask, dim=0)
    mask = torch.LongTensor(mask)
    return mask

from torch.utils.data.dataloader import DataLoader


class SciwingDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        """ This is an extension of the PyTorch DataLoader
        The collate function is always a list. The rest of the parameters
        can be sent by the user

        Parameters
        ----------
        dataset : Dataset
        batch_size : int
        shuffle
        sampler: torch.utils.data.Sampler
        batch_sampler
        num_workers: int
        drop_last : bool
        timeout : int
        worker_init_fn
        """
        super(SciwingDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            collate_fn=list,
            sampler=sampler,
        )

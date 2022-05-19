#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

class AverageMeter():
    """Computes and stores the average and current value."""

    def __init__(self,
                 name: str,
                 fmt: str = ":f") -> None:
        """Construct an AverageMeter module.

        :param name: Name of the metric to be tracked.
        :param fmt: Output format string.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset internal states."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update internal states given new values.

        :param val: New metric value.
        :param n: Step size for update.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Get string name of the object."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

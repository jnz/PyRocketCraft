# (c) Jan Zwiener (jan@zwiener.org)
#
# Basic interface for the different control algorithms.

class BaseControl:
    def __init__(self):
        pass

    def next(self, observation):
        """
        Should be overridden by all subclasses.
        Takes an observation, returns the next control input.
        """
        raise NotImplementedError

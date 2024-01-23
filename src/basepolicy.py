class BasePolicy:
    def __init__(self):
        pass

    def predict(self, observation):
        """
        Should be overridden by all subclasses.
        Takes an observation, returns the action.
        """
        raise NotImplementedError

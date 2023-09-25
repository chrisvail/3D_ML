


class Callback:
    """ Base class for callbacks for the Learner class. This should never be instantiated on it's own. 
    It wouldn't do anything if you did but it would be a waste of time. 

    Below is a list of all the methods that could be run by the Learner Class. Do note that each one has 2 variants "before_" and "after_" prepended to the name.
        - 
    
    
    """
    def __init__(self, order=0):
        self.order = order
    

class Frame(object):

    def __init__(self, screenshot):
        
        # The original data from the screenshot
        self.pixels = screenshot

    # Return a new Frame that has been grayscaled and scaled
    def preprocess(self):
        self.grayscale()
        self.scale()
        return self
 
    # Remove all color and leave only gray pixels
    # ***** Hopefully we don't have to do this since OpenGL can convert when doing glReadPixels! *****
    def grayscale(self):
        pass
    
    # Reduce the size of the given screenshot (2-D array of pixels)
    # ***** Hopefully we don't have to do this since the game supports resizing! *****
    def scale(self):
        pass
    

    # Return a 1-D list of grayscale pixel values for use in the CNN
    # If no other processing is required (not sure yet!) then just return the raw array of pixels
    def toCNNInput(self):
        return self.pixels


    def __str__(self):
        return "screenshot"








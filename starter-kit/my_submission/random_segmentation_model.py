import numpy as np

class RandomSegmentationModel:
    def __init__(self):
        """
        Initialize your model here
        """
        self.class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255]
    
    def segment_single_image(self, image_to_segment):
        """
        Implements a function to segment a single image
        Inputs:
            image_to_segment - Single frame from onboard the flight

        Outputs:
            An 2D image with the pixels values corresponding to the label number
        """
        image_size = image_to_segment.shape[:2]
        segmentation_results = np.random.choice(self.class_list, size=image_size)
        return segmentation_results
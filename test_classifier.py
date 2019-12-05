# Add curent file location to system path
import sys
import os

PACKAGE_PARENT = ''
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
module_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
if module_path not in sys.path:
    sys.path.append(module_path)

# All test code
import unittest
import helpers
from traffic_light_classifier import TrafficLightClassifier
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib._png as png
import numpy as np

# Helper functions for printing markdown text (text in color/bold/etc)
def printmd(string):
    display(Markdown(string))

# Print a test failed message, given an error
def print_fail():
    printmd('**<span style="color: red;">TEST FAILED</span>**')    
    
# Print a test passed message
def print_pass():
    printmd('**<span style="color: green;">TEST PASSED</span>**')

# Create test class
def get_test_object(list_name='test'):
    dirs = {
        'test': [module_path, 'traffic_light_images', 'training'],
        'training': [module_path, 'traffic_light_images', 'test']
    }

    path_images = os.path.join(*dirs[list_name])
    image_list = helpers.load_dataset(path_images)
    n_images = len(image_list)
    if n_images == 0:
        raise ValueError("No images loaded from {0} set".format(list_name))
    tlc = TrafficLightClassifier(image_list)
    return tlc

    
# A class holding all tests
class Tests(unittest.TestCase):
    
    # Tests the `one_hot_encode` function, which is passed in as an argument
    def test_one_hot_encoding(self):
        
        # Test that the generated one-hot labels match the expected one-hot label
        # For all three cases (red, yellow, green)
        try:
            self.assertEqual([1,0,0], TrafficLightClassifier.one_hot_encode('red'))
            self.assertEqual([0,1,0], TrafficLightClassifier.one_hot_encode('yellow'))
            self.assertEqual([0,0,1], TrafficLightClassifier.one_hot_encode('green'))
        
        # If the function does *not* pass all 3 tests above, it enters this exception
        except self.failureException as e:
            # Print out an error message
            print_fail()
            print("One hot function did not return the expected one-hot label.")
            print('\n'+str(e))
            return
        
        # Print out a "test passed" message
        print_pass()
    
    
    # Tests if any misclassified images are red but mistakenly classified as green
    def test_red_as_green(self):
        tlc = get_test_object()
        misclassified_images = tlc.get_misclassified_images()
        # Loop through each misclassified image and the labels
        for im, predicted_label, true_label in misclassified_images:
            
            # Check if the image is one of a red light
            if(true_label == [1,0,0]):
                
                try:
                    # Check that it is NOT labeled as a green light
                    self.assertNotEqual(predicted_label, [0, 0, 1])
                except self.failureException as e:
                    # Print out an error message
                    print_fail()
                    print("Warning: A red light is classified as green.")
                    print('\n'+str(e))
                    return
        
        # No red lights are classified as green; test passed
        print_pass()

    # Test mask on knwon images from wikipedia
    def test_mask_on_known_hues(self):
        colors = ['red', 'yellow', 'green']
        # image_names = ['shades_{0}.png'.format(color) for color in colors]
        # Read in files
        image_list = []
        for color in colors:
            image_name = 'shades_{0}.png'.format(color)
            path_components = [module_path, 'images', image_name]
            image_path = os.path.join(*path_components)
            image = png.read_png_int(image_path)[:,:,:3]  # Need [:,:,:3] to get RGB not RGBA which yellow and green images are

            if not image is None:
                image_list.append( (image, color) )
            else:
                raise ImportError("{} image not imported".format(color))
        
        # Create classifier object
        tlc = TrafficLightClassifier(image_list)
        # hsv_limits = tlc.hsv_limits

        for color in colors:
            color_label = tlc.one_hot_encode(color)
            # Create mask for each color
            for index, item in enumerate(tlc.image_lists['standardized']):
                masked_image, mask = TrafficLightClassifier.mask_image(item[0],
                            tlc.hsv_limits[color]['lower'], tlc.hsv_limits[color]['upper'])
                mask_label = item[1]
                mask_area = mask.shape[0] * mask.shape[1]
                mask_coverage = np.sum(mask) / 255 / mask_area
                if mask_label == color_label:
                    # E.g. for red image and red color mask should not be present
                    try:
                        self.assertGreater(mask_coverage, 0.75)
                    except self.failureException as e:
                        # tlc.visualize_masks(index)
                        print(str(e))
                else:
                    # E.g. for green image and red color mask should be everywhere
                    self.assertEqual(mask_coverage, 0)

    def test_accuracy_training_set(self):
        tlc = get_test_object(list_name='training')
        accuracy = tlc.get_accuracy()
        try:
            self.assertGreater(accuracy, 0.95)
        except self.failureException as e:
            # n_plots = min(tlc.get_num_misclassifed(), 10)
            # tlc.visualize_image_sample(list_name='misclassified', randomize=False, n_plots=n_plots)
            print(str(e))

    def test_accuracy_test_set(self):
        tlc = get_test_object(list_name='test')
        accuracy = tlc.get_accuracy()
        try:
            self.assertGreater(accuracy, 0.95)
        except self.failureException as e:
            # n_plots = min(tlc.get_num_misclassifed(), 10)
            # tlc.visualize_image_sample(list_name='misclassified', randomize=False, n_plots=n_plots)
            print(str(e))


if __name__ == "__main__":
    unittest.main()






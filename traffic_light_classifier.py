import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import math

class TrafficLightClassifier:
    """
    Classifies traffic light images as Red, Yellow or Green state

    Traffic lights in RGB format, cropped close are classified as being wither red, yellow or green

    Atrributes:
        image_lists: A dictionary containing lists of images (RGB numpy arrays) and labels (string).
            Format of [[np.array(..., ..., ...), label], ...].
            Dictionary includes:
                'original': Orginal images
                'standardized': Images standardized during pre-processing
        hsv_limits: A dictionary of lower and upper limits for HSV masking of red, yellow and green
            traffic light colors. Set by default but can be adjusted to improve classification.
    """

    def __init__(self, original_image_list):
        self.image_lists = {
            'original': None,
            'standardized': None,
            'masked': None,
            'masks': None
        }
        self.image_lists['original'] = original_image_list
        # Set HSV limits for maksing. Can be modified publically to improve classification
        self.set_default_hsv_limits()
        # Classify images
        self.classify_images()

    def set_default_hsv_limits(self):
        """ Sets default HSV limits for masking red, yellow and green colors"""
        s_limits = [50, 255]
        v_limits = [150, 255]

        self.hsv_limits = {
            'red': {
                'lower': [
                    np.array([0, s_limits[0], v_limits[0]]),
                    np.array([170, s_limits[0], v_limits[0]])
                ],
                'upper': [
                    np.array([10, s_limits[1], v_limits[1]]),
                    np.array([179, s_limits[1], v_limits[1]])
                ]
            },
            'yellow': {
                'lower': [ 
                    np.array([12, s_limits[0], v_limits[0]])
                ],
                'upper': [
                    np.array([45, s_limits[1], v_limits[1]])
                ]
            },  
            'green': {
                'lower': [
                    np.array([80, s_limits[0], v_limits[0]])
                ],
                'upper': [
                    np.array([114, s_limits[1], v_limits[1]])
                ]
            }
        }

    def visualize_image(self, image_num, list_name='original'):
        """
        Plots single image from defined set

        Args:
            image_num (int): Image index to plot
            list_name (string): Name of image list to select from
        """
        image = self.image_lists[list_name][image_num][0]
        label = self.image_lists[list_name][image_num][1]

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title('Image: {0}\nLabel: {1}\nShape: {2}'.format(image_num, label, image.shape))
        ax.set_xlabel('x pixels')
        ax.set_ylabel('y pixels')
        plt.show()

    def visualize_image_sample(self, list_name='original', n_plots=10, randomize=True):
        """
        Plots multiple images from defined set.

        Args:
            list_name (string): Name of image list to select from
            n_plots (int): Number of images to plot
            randomize (bool): Determines if images should be randomized or not
        """
        image_list = self.image_lists[list_name]

        max_cols = 10
        rows = math.ceil(n_plots / max_cols)
        cols = 10 if rows > 1 else n_plots
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2*rows))
        if randomize:
            training_subset = random.sample(image_list, n_plots)
        else:
            training_subset = image_list[:n_plots]
        for i, ax in zip(range(n_plots), axes.ravel()):
            image = training_subset[i][0]
            ax.imshow(image)
            ax.set_title('{0}\n{1}'.format(training_subset[i][1], image.shape),
                                fontdict={'fontsize': 10})
        plt.tight_layout()
        plt.show()

    @staticmethod
    def one_hot_encode(label):
        """
        One hot encodes 'red', 'yellow', 'green' traffic light tags.

        Examples: 
            one_hot_encode("red") will return: [1, 0, 0]
            one_hot_encode("yellow") will return: [0, 1, 0]
            one_hot_encode("green") will return: [0, 0, 1]

        Args:
            label (string): Traffic light color. Must be 'red', 'yellow' or 'green'

        Returns:
            one_hot_encoded (list): One hot encoded list from input
        """
        labels = ['red', 'yellow', 'green']
        if label not in labels:
            raise ValueError("label must be 'red', 'yellow' or 'green'")

        one_hot_encoded = [1 if label==x else 0 for x in labels] 
        
        return one_hot_encoded

    @staticmethod
    def standardize_image(image):
        """
        Preprosses image to standarize
        """
        standard_im = cv2.resize(np.copy(image), (25, 25))
        
        return standard_im

    def standardize_image_list(self, list_name='original'):
        """
        Standardizes image list with all relevent methods for images and labels.

        Args:
            list_name (string): Name of image list to select from.
        """
        # Create new list
        image_list_standard = []

        # Iterate through all the image-label pairs
        for item in self.image_lists[list_name]:
            image = item[0]
            label = item[1]
            # Standardize the image
            image_standard = self.standardize_image(image)
            # One-hot encode the label to standardize
            one_hot_label = self.one_hot_encode(label)    
            # Append the image, and it's one hot encoded label to the full, processed list of image data 
            image_list_standard.append((image_standard, one_hot_label))
            
        return image_list_standard

    @staticmethod
    def convert_image_to_hsv(rgb_image):
        """
        Converts RGB image to HSV format.

        Args:
            rgb_image: RGB image. Numpy array of shape (n_rows, n_cols, 3)

        Returns:
            hsv_image: HSV image. Numpy array of shape (n_rows, n_cols, 3)
        """
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        return hsv_image

    @staticmethod
    def mask_image(rgb_image, hsv_lowers, hsv_uppers):
        """
        Applies mask to image based upon hsv upper and lower limits.

        Args:
            rgb_image: RGB image. Numpy array of shape (n_rows, n_cols, 3).
            hsv_lower (list of numpy.array): List of numpy arrays in format (hue_lower, saturation_lower, value_lower).
            hsv_upper (list of numpy.array): List of numpy arrays in format (hue_upper, saturation_upper, value_upper).
        Returns:
            masked_image: Image in rgb format with mask applied. Numpy array of shape (n_rows, n_cols, 3).
            mask: 2D array with elements either 0 or 255 to represent mask. Numpy array of shape (n_rows, n_cols).
        """
        if len(hsv_lowers) != len(hsv_uppers):
            raise ValueError("hsv_lowers and hsv_uppers must be the same length")
        
        # Create hsv image copy
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Final mask to add to 
        final_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype='uint8')
        for hsv_lower, hsv_upper in zip(hsv_lowers, hsv_uppers):
            mask = cv2.inRange(hsv_image, hsv_lower, hsv_upper)
            final_mask = np.clip(final_mask + mask, 0, 255)
            
        # Mask rgb image
        masked_image = np.copy(rgb_image)
        
        masked_image[final_mask == 0] = [0,0,0]
        # Alternate method for applying mask 
    #     masked_image = cv2.bitwise_and(rgb_image,rgb_image, mask=final_mask)
        
        return masked_image, final_mask

    def mask_image_list(self, list_name='standardized', colors=None):
        """
        Applies mask to isolate red, yellow and green lights from all images

        Args:
            Args:
                list_name (string): Name of image list to select from.
                colors: List of strings to define colors from self.hsv_limits to mask.

        Returns:
            image_list_masked: A list of masked images (RGB numpy arrays) and labels (string).
        """
        if colors is None:
            colors = self.hsv_limits.keys()
        # Get specified images and labels
        images = [item[0] for item in self.image_lists[list_name]]
        labels = [item[1] for item in self.image_lists[list_name]]

        # Create list of all color limits
        lowers = []
        uppers = []
        for color in colors:
            lowers = lowers + self.hsv_limits[color]['lower']
            uppers = uppers + self.hsv_limits[color]['upper']
        
        image_list_masked = []
        image_list_masks = []
        for image, label in zip(images, labels):
            masked_image, mask = self.mask_image(image, lowers, uppers)
            image_list_masked.append( (masked_image, label) )
            image_list_masks.append( (mask, label) )

        return image_list_masked, image_list_masks

    @staticmethod
    def image_brightness(rgb_image):
        """
        Calculates average image brightness using HSV format value metric

        Args:
            rgb_image: RGB image. Numpy array of shape (n_rows, n_cols, 3).
            
        Returns:
            avg_brightness (float): Average brightness of image 
        """
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        # trim images to remove background at sides (generally lights are rectangles)
        hsv_image = hsv_image[2:-2, 4:-4, :]
        # Add up all the pixel values in the V channel
        sum_brightness = np.sum(hsv_image[:,:,2])
        
        # Average brightness for image
        avg_brightness = sum_brightness / (len(rgb_image) * len(rgb_image[0]))
        return avg_brightness

    def classify_image_by_brightness(self, image_index, list_name='masked'):
        """
        Classifies traffic light image by brightness

        Args:
            image_index (int): Index of image.
            list_name (string): Name of image list to select from.

        Returns:
            feature (list): One hot encoded feature classification
        """
        rgb_image = self.image_lists[list_name][image_index][0]
        colors = ['red', 'yellow', 'green']
        feature_brightness = []
        for color in colors:
            masked_image, _ = self.mask_image(rgb_image,
                                        self.hsv_limits[color]['lower'], self.hsv_limits[color]['upper'])
            feature_brightness.append(self.image_brightness(masked_image))
            
        max_brightness = max(feature_brightness)
        feature = [1 if x==max_brightness else 0 for x in feature_brightness]
        
        return feature

    def classify_image(self, image_index, list_name='masked'):
        """
        Classifies traffic light image by all available features

        Args:
            image_index (int): Index of image.
            list_name (string): Name of image list to select from.

        Returns:
            predicted_label (list): One hot encoded image classification
        """
        f_brightness = self.classify_image_by_brightness(image_index, list_name)
        predicted_label = f_brightness
        
        return predicted_label

    def get_misclassified_images(self):
        """
        Get misclassified images based upon labels.

        Returns:
            misclassified: List of misclassified images. Format: [image, predicted_label, true_label].
        """
        # Get specified images and labels
        images = [item[0] for item in self.image_lists['original']]
        labels = [item[1] for item in self.image_lists['standardized']]
        # Track misclassified images by placing them into a list
        misclassified = []
        indexes_misclassified = []  # Keep track for later access
        # Iterate through all the test images
        # Classify each image and compare to the true label
        for index, true_label in enumerate(labels):
            # Get true data
            assert(len(true_label) == 3), "The true_label {} is not the expected length (3).".format(true_label)
            # Get predicted label from your classifier
            predicted_label = self.classify_image(index)
            assert(len(predicted_label) == 3), "The predicted_label {} is not the expected length (3).".format(predicted_label)
            # Compare true and predicted labels 
            if(predicted_label != true_label):
                # If these labels are not equal, the image has been misclassified
                misclassified.append((images[index], predicted_label, true_label))
                indexes_misclassified.append(index)
        # Return the list of misclassified [image, predicted_label, true_label] values
        self.indexes_misclassified = indexes_misclassified  # Save as private attribute
        return misclassified

    def get_misclassified_masks(self):
        """
        Gets misclassified mask images based upon misclassified image set

        Returns:
            misclassified_masks: List of misclassified masks in format, images (RGB numpy arrays) and labels (string).
        """
        misclassified_masks = []
        for index in self.indexes_misclassified:
            misclassified_masks.append(self.image_lists['masks'][index])
        return misclassified_masks

    def classify_images(self):
        """
        Runs all necessary methods to classify images.

        Must be re-called if attributes are changed, e.g.hsv_limits.
        """
        self.image_lists['standardized'] = self.standardize_image_list()
        masked, masks = self.mask_image_list()
        self.image_lists['masked'] = masked
        self.image_lists['masks'] = masks
        self.image_lists['misclassified'] = self.get_misclassified_images()

    def get_num_misclassifed(self):
        """
        Gets number of misclassified images

        Returns:
            (int): Number of misclassified images
        """
        return len(self.image_lists['misclassified'])

    def get_accuracy(self):
        """
        Gets accuracy of classifiaction as ratio.
        
        Returns:
            (float): Ratio of misclassified images to tested images
        """
        return self.get_num_misclassifed() / len(self.image_lists['original'])

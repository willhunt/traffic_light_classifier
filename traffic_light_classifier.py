import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import optimize, stats
import random
import cv2
import math
import copy

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
    ERROR_MESSAGES = {
        'color_arg': "Argument color must be 'red', 'yellow' or 'green'"
    }

    def __init__(self, original_image_list=None):
        self.image_lists = {
            'original': None,
            'standardized': None,
            'masked': None,
            'masks': None
        }
        # Set HSV limits for maksing. Can be modified publically to improve classification
        self.set_default_hsv_limits()
        # Set masksize sigmoid function values
        self.set_default_masksize_sigmoid_values()

        if not original_image_list is None:
            self.image_lists['original'] = original_image_list
            # Classify images
            self.classify_images()

    def set_original_image_list(self, original_image_list):
        """ Sets original image list. Also runs preprocessing and classification steps """
        self.image_lists['original'] = original_image_list
        self.classify_images()

    def set_default_hsv_limits(self):
        """ Sets default HSV limits for masking red, yellow and green colors"""
        s_limits = [25, 255] #35
        v_limits = [12, 255]  #125

        self.hsv_limits = {
            'red': {
                'lower': [
                    np.array([1, s_limits[0], v_limits[0]]),  # h=1
                    np.array([157, s_limits[0], v_limits[0]])  # h=172
                ],
                'upper': [
                    np.array([6, s_limits[1], v_limits[1]]),  # h=5
                    np.array([220, s_limits[1], v_limits[1]])  # h=185
                ]
            },
            'yellow': {
                'lower': [ 
                    np.array([10, s_limits[0], v_limits[0]])  # h=13
                ],
                'upper': [
                    np.array([80, s_limits[1], v_limits[1]])  # h=40
                ]
            },  
            'green': {
                'lower': [
                    np.array([50, s_limits[0], v_limits[0]]) # h=75
                ],
                'upper': [
                    np.array([102, s_limits[1], v_limits[1]]) # h=93
                ]
            }
        }
    
    def get_hsv_range_for_optimisation(self, hsv):
        ranges = {}

        ranges['h'] = {
            'red': {
                'lower': [ [0, 2], [150, 180] ],
                'upper': [ [3, 15], [170, 220] ]
            },
            'yellow': {
                'lower': [ [5, 40] ],
                'upper': [ [41, 80] ]
            },  
            'green': {
                'lower': [ [20, 100] ],
                'upper': [ [85, 110] ]
            }
        }
        # Saturation range
        range_s = {'lower': [20, 100], 'upper': [254, 255]}
        ranges['s'] = copy.deepcopy(ranges['h'])
        for color in ranges['s'].keys():
            for limit in ranges['s'][color].keys():
                for s_range in ranges['s'][color][limit]:
                    s_range[:] = range_s[limit][:]
        # Value range
        range_v = {'lower': [100, 150], 'upper': [254, 255]}
        ranges['v'] = copy.deepcopy(ranges['s'])
        for color in ranges['v'].keys():
            for limit in ranges['v'][color].keys():
                for v_range in ranges['v'][color][limit]:
                    v_range[:] = range_v[limit][:]
        return ranges[hsv]

    def set_default_masksize_sigmoid_values(self):
        self.masksize_sigmoid_values = {
            'mid': 0.1,
            'scale': -100,
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

    def visualize_image_sample(self, list_name='original', n_plots=7, randomize=True):
        """
        Plots multiple images from defined set.

        Args:
            list_name (string): Name of image list to select from
            n_plots (int): Number of images to plot
            randomize (bool): Determines if images should be randomized or not
        """
        image_list = self.image_lists[list_name]

        max_cols = 7
        rows = math.ceil(n_plots / max_cols)
        cols = max_cols if rows > 1 else n_plots
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
    def one_hot_encode(color):
        """
        One hot encodes 'red', 'yellow', 'green' traffic light tags.

        Examples: 
            one_hot_encode("red") will return: [1, 0, 0]
            one_hot_encode("yellow") will return: [0, 1, 0]
            one_hot_encode("green") will return: [0, 0, 1]

        Args:
            color (string): Traffic light color. Must be 'red', 'yellow' or 'green'

        Returns:
            one_hot_encoded (list): One hot encoded list from input
        """
        colors = ['red', 'yellow', 'green']
        if color not in colors:
            raise ValueError(TrafficLightClassifier.ERROR_MESSAGES['color_arg'])

        one_hot_encoded = [1 if color==x else 0 for x in colors] 
        
        return one_hot_encoded

    @staticmethod
    def change_brightness_and_contrast(rgb_image, alpha, beta):
        """
        Changes brightness and contrast of an RGB image

        Args:
            alpha (float): contrast adjustment
            beta (float): brightness adjustment
        """
        changed_image = cv2.convertScaleAbs(rgb_image, alpha=alpha, beta=beta)
        return changed_image

    @staticmethod
    def standardize_image(rgb_image):
        """
        Preprosses image to standarize
        """
        # Standardize size
        square_image = cv2.resize(np.copy(rgb_image), (25, 25))

        # Standardize contrast - didn't help in the end
        # YUV VERSION - helps a tiny bit but throws the hues off
        # stdcontrast_image =  cv2.cvtColor(square_image, cv2.COLOR_RGB2YUV)
        # # Equlaize histogram on value channel
        # stdcontrast_image[:,:,0] = cv2.equalizeHist(stdcontrast_image[:,:,0])
        # #  Convert back to RGB
        # stdcontrast_image = cv2.cvtColor(stdcontrast_image, cv2.COLOR_YUV2RGB)

        # Increase contrast - didn't help in the end
        # highcontrast_image = TrafficLightClassifier.change_brightness_and_contrast(square_image, 2.0, -20.0)
        
        return square_image

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
        hsv_image = hsv_image[2:-2, 5:-5, :]
        # Add up all the pixel values in the V channel
        sum_brightness = np.sum(hsv_image[:,:,2])
        
        # Average brightness for image
        avg_brightness = sum_brightness / (len(rgb_image) * len(rgb_image[0]))
        return avg_brightness

    def classify_image_by_brightness(self, image_index, list_name='standardized'):
        """
        Classifies traffic light image by brightness

        Args:
            image_index (int): Index of image.
            list_name (string): Name of image list to select from.

        Returns:
            p_brightness (list): Probabilities of that light is showing each color in one hot encoded list order.
        """
        rgb_image = self.image_lists[list_name][image_index][0]
        colors = ['red', 'yellow', 'green']
        brightness = []
        for color in colors:
            masked_image, _ = self.mask_image(rgb_image,
                                        self.hsv_limits[color]['lower'], self.hsv_limits[color]['upper'])
            brightness.append(self.image_brightness(masked_image))
        
        # Scale brightness by maximum to create probability like metric
        max_brightness = max(brightness)
        p_brightness = [0 if max_brightness==0 else x / max_brightness for x in brightness]
        
        return p_brightness

    @staticmethod
    def mask_coverage(mask):
        """
        Calculates coverage of mask as a proportion of mask size.

        Args:
            mask: Numpt array of shape (n_rows, n_cols).
        Returns:
            mask_coverage (float): mask coverage ratio, between 0 and 1.
        """
        mask_area = mask.shape[0] * mask.shape[1]
        mask_coverage = np.sum(mask) / 255 / mask_area
        return mask_coverage

    @staticmethod
    def sigmoid(X, x_scale=1, x_shift=0):
        X_new = (X + x_shift) * x_scale 
        return 1 / (1 + np.exp(-X_new))

    def classify_image_by_mask_size(self, image_index, list_name='standardized'):
        """
        Classifies traffic light image by size of color mask. Traffic light bulb are only a small proportion of whole light.

        Args:
            image_index (int): Index of image.
            list_name (string): Name of image list to select from.

        Returns:
            p_masksize (list): Probabilities of that light is showing each color in one hot encoded list order.
        """
        rgb_image = self.image_lists[list_name][image_index][0]
        colors = ['red', 'yellow', 'green']
        coverages = []
        n_zero_coverage = 0
        for color in colors:
            _, mask = self.mask_image(rgb_image,
                                        self.hsv_limits[color]['lower'], self.hsv_limits[color]['upper'])
            coverage = self.mask_coverage(mask)
            coverages.append(coverage)
            if coverage == 0:
                n_zero_coverage += 1
        
        if n_zero_coverage == 2:
            p_masksize = [1 if x>0 else 0 for x in coverages]
        elif n_zero_coverage == 3:
            p_masksize = [1, 1, 1]
        else:
            # Apply sigmoid function to coverage to weight masks of smaller size with higher probability
            mid = self.masksize_sigmoid_values['mid']
            scale = self.masksize_sigmoid_values['scale']
            p_masksize = [self.sigmoid(x, scale, mid) for x in coverages]
        
        return p_masksize

    @staticmethod
    def get_probability_from_position(mask, color):
        """
        Calculates probability that a light is a certain color based upon the location of the image mask pixels

        Args:
            mask (numpy.array): Image mask as 2D numpy array
            color (string): Traffic light color

        """
        n_rows, n_cols = mask.shape

        colors = ['red', 'yellow', 'green']
        if color not in colors:
            raise ValueError(TrafficLightClassifier.ERROR_MESSAGES['color_arg'])

        i_color = colors.index(color)
        # Mean in centerhorizontally and vertically positioned where light is expected
        mu = [i_color * n_rows / 4, n_cols / 2]
        # Covariance
        cov_scale = 100
        cov = cov = np.array([[cov_scale, 0],[0, cov_scale]])
        # Multivariate normal distribution used to determine likelyhood of position
        rv = stats.multivariate_normal(mu, cov)

        p_pos = 0
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                # Get cumulative probability for each point from 2D probability density function
                if mask[i_row, i_col] > 0:
                    p_pos += rv.pdf([i_row, i_col])
        return p_pos

    def classify_image_by_mask_position(self, image_index, list_name='standardized'):
        """
        Classifies traffic light image by size of color mask. Traffic light bulb are only a small proportion of whole light.

        Args:
            image_index (int): Index of image.
            list_name (string): Name of image list to select from.

        Returns:
            p_maskpos (list): Probabilities of that light is showing each color in one hot encoded list order.
        """
        rgb_image = self.image_lists[list_name][image_index][0]
        colors = ['red', 'yellow', 'green']
        p_maskpos = []

        for color in colors:
            _, mask = self.mask_image(rgb_image,
                                        self.hsv_limits[color]['lower'], self.hsv_limits[color]['upper'])
            p_maskpos.append(self.get_probability_from_position(mask, color))
        # Normalize probabilties to 1 for maximum
        p_max = max(p_maskpos)
        if p_max > 0:
            p_maskpos = [x / p_max for x in p_maskpos]
        else:
            p_maskpos = [1, 1, 1]
        
        return p_maskpos

    def classify_image(self, image_index, list_name='standardized', methods=['brightness', 'maskpos']):
        """
        Classifies traffic light image by all available features

        Args:
            image_index (int): Index of image.
            list_name (string): Name of image list to select from.
            methods (list): List of methods to use for classification

        Returns:
            predicted_label (list): One hot encoded image classification
        """
        if type(methods) != list:
            raise ValueError("methods argument must be a list")
        classification_functions = {
            'brightness': self.classify_image_by_brightness,
            'masksize': self.classify_image_by_mask_size,
            'maskpos': self.classify_image_by_mask_position
        }
        p_combined = [1, 1, 1]
        for method in methods:
            p_method = classification_functions[method](image_index, list_name)
            p_combined = [x*y for x,y in zip(p_combined, p_method)]
        # Bias for red light over green
        bias = 0.25
        if (p_combined[0] > 0) and (p_combined[2] > 0):
            p_combined[0] *= (1 + bias)
            p_combined[2] *= (1 - bias)

        # Convert to one hot encoded prediction
        p_max = max(p_combined)
        predicted_label = [1 if x==p_max else 0 for x in p_combined]
        # Default to red if unsure as it is safer
        if sum(predicted_label) > 1:
            predicted_label = [1, 0, 0]
        
        return predicted_label

    def get_misclassified_images(self, methods=['brightness', 'maskpos']):
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
            predicted_label = self.classify_image(index, methods=methods)
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

    def visualize_masks(self, image_num, list_name='standardized', colors=None, viz_type='mask', standardize=True):
        """
        Visualizes the red yellow and green masks together for an image
        """
        if colors is None:
            colors = self.hsv_limits.keys()

        fig, axes = plt.subplots(1, len(colors))
        fig.suptitle("Mask for image {0} from list {1}".format(image_num, list_name))
        rgb_image = self.image_lists[list_name][image_num][0]
        if standardize:
            rgb_image = self.standardize_image(rgb_image)
        for color, ax in zip(colors, axes):
            ax.set_title(color)
            masked_image, mask = self.mask_image(rgb_image,
                                        self.hsv_limits[color]['lower'], self.hsv_limits[color]['upper'])
            if viz_type == 'mask':
                ax.imshow(mask, cmap='gray')
            elif viz_type == 'image':
                ax.imshow(masked_image, cmap='gray')
            else:
                raise ValueError("Argument viz_type must be either 'mask' or 'image'")
        plt.show()

    def classify_images(self, methods=['brightness', 'maskpos']):
        """
        Runs all necessary methods to classify images.

        Must be re-called if attributes are changed, e.g.hsv_limits.
        """
        self.image_lists['standardized'] = self.standardize_image_list()
        masked, masks = self.mask_image_list()
        self.image_lists['masked'] = masked
        self.image_lists['masks'] = masks
        self.image_lists['misclassified'] = self.get_misclassified_images(methods=methods)

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
        return 1 - self.get_num_misclassifed() / len(self.image_lists['original'])

    def set_lower_sv_thesholds(self, s_lower, v_lower):
        """
        Sets lower saturation and value thresholds for all colors for HSV masking.

        Args:
            s_lower (float): Lower saturation threshold.
            v_lower (float): Lower value threshold
        """
        for color in self.hsv_limits.keys():
            for threshold in self.hsv_limits[color]['lower']:
                if not s_lower is None:
                    threshold[1] = s_lower
                if not v_lower is None:
                    threshold[2] = v_lower
    
    def set_hsv_thresholds(self, hsv, color, limit, h_new, limit_index=0):
        """
         Sets hue thresholds for all colors for HSV masking.

        Args:
            hsv (string): 'h', 's', or 'v'
            color (string): 'Red', 'yellow', or 'green' color.
            limit (string): 'upper' or 'lower' limit'
            h_new (string): New hue threshold value
        """
        if hsv not in ['h', 's', 'v']:
            raise ValueError("hsv argument must be 'h', 's', or 'v'")
        if limit not in ['upper', 'lower']:
            raise ValueError("limit argument must be either 'lower' or 'upper'")
        colors = ['red', 'yellow', 'green']
        if color not in colors:
            raise ValueError(TrafficLightClassifier.ERROR_MESSAGES['color_arg'])

        hash_hsv = {'h':0, 's':1, 'v':2}
        self.hsv_limits[color][limit][limit_index][hash_hsv[hsv]] = h_new

    
    def train_classifier(self, training_image_list):
        """
        Train threshold values in classifier using a training data set.

        Args:
            training_image_list (list): List of images and labels for training thresholds.
        
        Returns:
            results (dict): Dictionary of results
        """
        # Create new classifier object
        tlc_training = TrafficLightClassifier(training_image_list)

        def train_for_hsv(tlc, hsv):
            """
            Args:
                tlc (TrafficLightClassifier)
                hsv (char): 'h', 's' or 'v'
            """
            ranges = tlc.get_hsv_range_for_optimisation(hsv)
            results = {}
            for color in ranges.keys():
                for r_i, (low, high) in enumerate(zip(ranges[color]['lower'], ranges[color]['upper'])):
                    color_range = "{0} ({1})".format(color, r_i)
                    results[color_range] = {}
                    for limit_name, limit_range in zip(['lower', 'upper'], [low, high]):
                        # Set other limit to closest end so independant optimisation has the best chance
                        if limit_name == 'lower':
                            self.set_hsv_thresholds(hsv, color, 'upper', high[0], r_i)
                        else:
                            self.set_hsv_thresholds(hsv, color, 'lower', low[1], r_i)
                        result = optimize.minimize_scalar(
                            tlc.change_and_evaluate_hsv_thresholds,
                            bounds=limit_range,
                            method='bounded',
                            args=(hsv, color, limit_name, r_i)
                        )
                        results[color_range][limit_name] = result.x

                    # Set this optimum as new value (once both upper and lower are optimised)
                    self.set_hsv_thresholds(hsv, color, 'lower', results[color_range]['lower'], r_i)
                    self.set_hsv_thresholds(hsv, color, 'upper', results[color_range]['upper'], r_i)

                    print("\tMinimum for {0} found with {1} range ({2:.1f}, {3:.1f})".format(color_range, hsv.capitalize(),
                            results[color_range]['lower'], results[color_range]['upper']))
            print("Training accuracy at {0:.2f} % after optimising {1} values".format(tlc.get_accuracy() * 100, hsv.capitalize()))
            return results

        # Train HSV values
        print("Initial/default training accuracy at {0:.2f} %".format(tlc_training.get_accuracy() * 100))
        results = {}
        for hsv in ['v', 's', 'h']:
            results[hsv] = train_for_hsv(tlc_training, hsv)

        print("Final training accuracy at {0:.1f} %".format(tlc_training.get_accuracy() * 100))
        # Now update this object with new values
        self.hsv_limits = tlc_training.hsv_limits
        self.classify_images()
        print("Actual (test) at {0:.1f} %".format(self.get_accuracy() * 100))
        return results

    def change_and_evaluate_hsv_thresholds(self, new_threshold, hsv, color, limit, limit_index=0):
        """
        Function for optimizer to changehsv thresholds then classify and return 1 - accuracy (optimiser minimizes this)
        """
        colors = ['red', 'yellow', 'green']
        if color not in colors:
            raise ValueError(TrafficLightClassifier.ERROR_MESSAGES['color_arg'])

        self.set_hsv_thresholds(hsv, color, limit, new_threshold, limit_index)
        self.classify_images()
        accuracy = self.get_accuracy()
        # Add in penalty for scoring red light as green - NB: This doesn't seem to work
        # n_red_as_green = self.get_num_red_light_classified_as_green()
        # if n_red_as_green > 0:
        #     accuracy = 0
        # accuracy -= n_red_as_green * 0.  # Penalize
        return 1 - max(accuracy, 0)

    def get_num_red_light_classified_as_green(self):
        """
        Returns number of red lights that have been misclassified as green lights

        Returns:
            n_red_as_green (int): Number of red lights that have been classified as green lights
        """
        n_red_as_green = 0
        for _, predicted_label, true_label in self.image_lists['misclassified']:
            if true_label == [1, 0, 0]:
                if predicted_label == [0, 0, 1]:
                    n_red_as_green += 1
        return n_red_as_green

        
    def plot_effect_of_sv_thresholds_surf(self):
        """
        Plots effect of lower saturation and value thresholds using brightness classification method. Ploted with surface to view interactions.
        """
        # Get accuracy over range of SV lower limits
        n = 5
        s_lowers = np.linspace(0, 254, num=n)
        v_lowers = np.linspace(0, 254, num=n)
        xyz_results = []
        for s_i in range(n):
            for v_i in range(n):
                self.set_lower_sv_thesholds(s_lowers[v_i], v_lowers[s_i])
                self.classify_images(methods=['brightness'])
                xyz_results.append( (s_lowers[s_i], v_lowers[v_i], self.get_accuracy()) )
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # Extract coordinates
        xs = [a[0] for a in xyz_results]
        ys = [a[1] for a in xyz_results]
        zs = [a[2] for a in xyz_results]
        # Plot points
        ax.scatter(xs, ys, zs, label='True Position')
        # Add data labels
        max_accuracy = max(zs)
        for x,y,z in zip(xs, ys, zs):
            if z == max_accuracy:
                text = "({0:.0f}, {1:.0f}, {2:.2f})".format(x, y, z)
                ax.text(x, y, z, text, zdir=(1, 1, 0))
        # Plot the surface.
        ax.plot_trisurf(xs, ys, zs, cmap=cm.coolwarm)
        ax.set_xlabel('Saturation Lower Threshold')
        ax.set_ylabel('Value Lower Threshold')
        ax.set_zlabel('Classifier Accuracy')
        fig.suptitle('Effect of HSV Saturation and Value Lower Thresholds\nOn Traffic Light Classifier Accuracy')
        plt.show()
        # Reset hsv limit
        self.set_default_hsv_limits()

    def plot_effect_of_sv_thresholds(self, methods=[ ['brightness'], ['brightness', 'masksize'], ['maskpos'], ['brightness', 'maskpos'] ]):
        """
        Plots effect of lower saturation and value thresholds using all classification methods. Values evaluated seperately without interaction.

        Returns:
            results: Dictionary of results.
        """
        n = 4
        ranges = {
            'saturation': np.linspace(0, 254, num=n),
            'value': np.linspace(0, 254, num=n)
        }
        results = {}
        for variable in ranges.keys():
            results[variable] = {}
            for method in methods:
                method_str = str(method)
                results[variable][method_str] = []
                for value in ranges[variable]:
                    if variable == 'saturation': 
                        self.set_lower_sv_thesholds(value, None)
                    else:
                        self.set_lower_sv_thesholds(None, value)
                    self.classify_images(methods=method)
                    results[variable][method_str].append( (value, self.get_accuracy()) )
            self.set_default_hsv_limits()

        # Plot
        plot_cols = len(results)
        fig = plt.figure()
        plot_i = 1
        for variable in results.keys():
            # Plot the results for this color
            ax = fig.add_subplot(1, plot_cols, plot_i)
            for method in methods:
                method_str = str(method)
                x = [a[0] for a in results[variable][method_str]]
                y = [a[1] for a in results[variable][method_str]]
                ax.plot(x, y, label=method_str)

            ax.set_xlabel('Value')
            ax.set_ylabel('Classifier Accuracy')
            ax.set_title( "{0} Lower Threshold".format(variable.capitalize()) )
            plot_i += 1

        # fig.suptitle('Effect of Saturation & Value Lower Thresholds On Traffic Light Classifier Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return results

    def plot_effect_of_masksize_sigmoid(self):
        """
        Plots effect of masksize sigmoid parameters with all (combined) classification methods. Ploted with surface to view interactions.
        """
        n = 5
        mid_range = np.linspace(0, 1.2, num=n)
        scale_range = np.linspace(-100, -25, num=n)
        xyz_results = []
        for m_i in range(n):
            self.masksize_sigmoid_values['mid'] = mid_range[m_i]
            for s_i in range(n):
                self.masksize_sigmoid_values['scale'] = scale_range[s_i]
                self.classify_images(methods=['masksize', 'brightness'])
                xyz_results.append( (mid_range[m_i], scale_range[s_i], self.get_accuracy()) )

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # Extract coordinates
        xs = [a[0] for a in xyz_results]
        ys = [a[1] for a in xyz_results]
        zs = [a[2] for a in xyz_results]
        # Plot points
        ax.scatter(xs, ys, zs, label='True Position')
        # Add data labels
        max_accuracy = max(zs)
        for x,y,z in zip(xs, ys, zs):
            if z == max_accuracy:
                text = "({0:.3f}, {1:.3f}, {2:.2f})".format(x, y, z)
                ax.text(x, y, z, text, zdir=(1, 1, 0))
        # Plot the surface.
        ax.plot_trisurf(xs, ys, zs, cmap=cm.coolwarm)

        ax.set_xlabel('Mean')
        ax.set_ylabel('Scale')
        ax.set_zlabel('Classifier Accuracy')
        # fig.suptitle('Effect of Mask Size Sigmoid Values\nOn Traffic Light Classifier Accuracy')
        plt.show()
        # Reset pdf values
        self.set_default_masksize_sigmoid_values()

    def plot_effect_of_hue_thresholds(self, methods=[ ['brightness'], ['brightness', 'masksize'], ['maskpos'], ['brightness', 'maskpos'] ]):
        """
        Plots effect of hue upper and lower thresholds for all classification methods. Values evaluated seperately without interaction.

        Returns:
            results: Dictionary of results.
        """
        n = 5
        ranges = self.get_hsv_range_for_optimisation('h')
        results = {}
        for color in ranges.keys():
            for r_i, (low, high) in enumerate(zip(ranges[color]['lower'], ranges[color]['upper'])):
                low_range = np.linspace(low[0], low[1], num=n)
                high_range = np.linspace(high[0], high[1], num=n)
                color_range = "{0} ({1})".format(color, r_i)
                results[color_range] = {}
                for method in methods:
                    method_str = str(method)
                    results[color_range][method_str] = {'lower': [], 'upper': []}
                    for low_hue in low_range:
                        self.hsv_limits[color]['lower'][r_i][0] = low_hue
                        self.classify_images(methods=method)
                        results[color_range][method_str]['lower'].append( (low_hue, self.get_accuracy()) )
                    self.set_default_hsv_limits()
                    for high_hue in high_range:    
                        self.hsv_limits[color]['upper'][r_i][0] = high_hue
                        self.classify_images(methods=method)
                        results[color_range][method_str]['upper'].append( (high_hue, self.get_accuracy()) )
                    self.set_default_hsv_limits()
                
            
        # Plot
        plot_cols = len(results)
        plot_rows = 2
        fig = plt.figure(figsize=(12, 6))
        plot_i = 1
        for threshold in ['lower', 'upper']:
            for color_range in results.keys():
                # Plot the results for this color
                ax = fig.add_subplot(plot_rows, plot_cols, plot_i)
                for method in methods:
                    method_str = str(method)
                    x = [a[0] for a in results[color_range][method_str][threshold]]
                    y = [a[1] for a in results[color_range][method_str][threshold]]
                    ax.plot(x, y, label=method_str)

                ax.set_xlabel('Hue Value')
                ax.set_ylabel('Classifier Accuracy')
                ax.set_title( "{0} {1} Hue Threshold".format(threshold.capitalize(), color_range.capitalize()) )
                plot_i += 1

        # fig.suptitle('Effect of Hue Thresholds On Traffic Light Classifier Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return results

#%% Add curent file location to system path
import sys
import os

PACKAGE_PARENT = ''
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
module_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
import helpers
from test_classifier import Tests
from traffic_light_classifier import TrafficLightClassifier

# %% Load stuff
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"
dirs = [IMAGE_DIR_TRAINING, IMAGE_DIR_TEST]
keys = ['training', 'test']

IMAGE_LIST = {}
for directory, key in zip(dirs, keys):
    path_images = os.path.join(module_path, directory)
    IMAGE_LIST[key] = helpers.load_dataset(path_images)
    n_images = len(IMAGE_LIST[key])
    if n_images == 0:
        print("No images loaded from {0} set".format(key))
    else:
        print("Loaded {0} images from {1} set".format(n_images, key))


# %% Create Object
tlc = TrafficLightClassifier(IMAGE_LIST['test'])
tlc_training = TrafficLightClassifier(IMAGE_LIST['training'])

# %% Look at HSV thresholds
# tlc_training.plot_effect_of_sv_thresholds()

# %% Look at sigmoid function effect
# tlc_training.plot_effect_of_masksize_sigmoid()

# %% Look at hue ranges
tlc_training.plot_effect_of_hue_thresholds()

# %%
# tlc.train_classifier(IMAGE_LIST['training'])





SAVE_PATH_CAPTURE = 'C:\\Users\\compops\\Desktop\\David_Folder\\gated_captures'
SAVE_PATH_SINGLE_PIXEL = 'C:\\Users\\compops\\Desktop\\David_Folder\\gated_single_pixel'
SAVE_PATH_CORRELATIONS = "C:\\Users\\compops\\Desktop\\David_Folder\\gated_correlations"

READ_PATH_CORRELATIONS_WINDOWS = f'C:\\Users\\compops\\Desktop\\David_Folder\\gated_correlations'
READ_PATH_CORRELATIONS_MAC = f'/Users/davidparra/PycharmProjects/py-gated-camera/data/correlation_functions'

# Single-pixel experiments read correlations from their own subfolder ...
READ_PATH_CORRELATIONS_SINGLE_PIXEL_WINDOWS = f'C:\\Users\\compops\\Desktop\\David_Folder\\gated_correlations\\single_pixel'
READ_PATH_CORRELATIONS_SINGLE_PIXEL_MAC = f'/Users/davidparra/PycharmProjects/py-gated-camera/data/correlation_functions/single_pixel'

# ... while depth decoding reads from a separate subfolder.
READ_PATH_CORRELATIONS_DEPTH_WINDOWS = f'C:\\Users\\compops\\Desktop\\David_Folder\\gated_correlations\\depth_decoding'
READ_PATH_CORRELATIONS_DEPTH_MAC = f'/Users/davidparra/PycharmProjects/py-gated-camera/data/correlation_functions/depth_decoding'

READ_PATH_VOLTAGE_FUNCTIONS_WINDOWS = "/depreciated/voltage_functions"
READ_PATH_VOLTAGE_FUNCTIONS_MAC = "/Users/davidparra/PycharmProjects/py-gated-camera/voltage_functions"

READ_PATH_CAPTURE_WINDOWS = "C:\\Users\\compops\\Desktop\\David_Folder\\gated_captures"
READ_PATH_CAPTURE_MAC = "/Users/davidparra/PycharmProjects/py-gated-camera/data/capture"

READ_PATH_SINGLE_PIXEL_WINDOWS = "C:\\Users\\compops\\Desktop\\David_Folder\\gated_single_pixel"
READ_PATH_SINGLE_PIXEL_MAC = "/Users/davidparra/PycharmProjects/py-gated-camera/data/single_pixel"

HOT_MASK_PATH_WINDOWS = 'C:\\Users\\compops\\PycharmProjects\\py-gated-camera\\masks\\hot_mask1.npy'
HOT_MASK_PATH_MAC = '/Users/davidparra/PycharmProjects/py-gated-camera/masks/hot_mask1.npy'


### GlobaL constants for SPAD 512
PORT = 9999  # Check the command Server in the setting tab of the software and change it if necessary
VEX = 7

EPILSON = 1e-8
SPEED_OF_LIGHT = 3e8

PIXEL_PITCH = 16.38 #in uM
FOCAL_LENGTH = 25 #in mm

# SINGLE_PIXEL_COORDS = {'x': [85, 105],
#                        'y': [185, 205]}

# SINGLE_PIXEL_COORDS = {'x': [65, 125],
#                        'y': [165, 225]}


# Single-pixel ROI, keyed by im_width. The sensor height is always 512, but the
# width follows im_width, so the x window has to move with it (the target sits
# at a different column when the frame is cropped).
# SINGLE_PIXEL_COORDS_BY_WIDTH = {
#     128: {'x': [15, 35],   'y': [255, 285]},
#     512: {'x': [115, 135], 'y': [250, 280]},
# }


SINGLE_PIXEL_COORDS_BY_WIDTH = {
    128: {'x': [20, 30],   'y': [260, 280]},
    512: {'x': [210, 220], 'y': [260, 280]},
}

# default so existing `from utils.global_constants import SINGLE_PIXEL_COORDS`
# imports keep working; prefer get_single_pixel_coords(im_width) where im_width
# is known at runtime.
SINGLE_PIXEL_COORDS = SINGLE_PIXEL_COORDS_BY_WIDTH[128]

# older ROIs kept for reference
# SINGLE_PIXEL_COORDS = {'x': [20, 30],
#                        'y': [250, 280]}


def get_single_pixel_coords(im_width=None):
    """ROI dict for a given im_width, falling back to the 128 default."""
    if im_width is None:
        return SINGLE_PIXEL_COORDS
    return SINGLE_PIXEL_COORDS_BY_WIDTH.get(int(im_width), SINGLE_PIXEL_COORDS)


def get_total_pixels(coords=None):
    """Pixel count of an ROI dict (defaults to the module default)."""
    c = SINGLE_PIXEL_COORDS if coords is None else coords
    return (c['y'][1] - c['y'][0]) * (c['x'][1] - c['x'][0])
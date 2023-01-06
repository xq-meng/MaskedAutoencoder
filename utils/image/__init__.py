from utils.image.tensor2PIL import tensor2PIL


def is_image(filename: str):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
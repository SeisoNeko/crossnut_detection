import os

def upscale_image(input_path, output_dir, options='', model='RealESRGAN_x4plus'):
    """
    Upscales an image using Real-ESRGAN.
    
    Parameters:
    - input_path: Path to the input image.
    - output_dir: Path to save the upscaled image.
    """
    path = os.path.join(os.getcwd(), 'Real_ESRGAN')
    os.chdir(path)
    os.system('python inference_realesrgan.py -n {} -i {} -o {} {}'.format(model, input_path, output_dir, options))

if __name__ == '__main__':

    input_image_path = r'D:\Code\road_excavation_detect\4.jpg'  # Path to the input image
    output_image_dir = r'D:\Code\road_excavation_detect\Real_ESRGAN\output'  # Path to save the upscaled image
    upscale_image(input_image_path, output_image_dir)
    
import os

def upscale_image(input_path, output_dir, options='', model='RealESRGAN_x4plus'):
    """
    Upscales an image using Real-ESRGAN.
    
    Parameters:
    - input_path: Path to the input image.
    - output_dir: Path to save the upscaled image.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    path = os.path.join(os.getcwd(), 'Real_ESRGAN')
    if os.getcwd() != path:
        try:
            os.chdir(path)
        except FileNotFoundError: 
            pass
    os.system(f'python inference_realesrgan.py -n {model} -i {input_path} -o {output_dir} {options}')

if __name__ == '__main__':

    input_image_path = r'D:\Code\road_excavation_detect\data\test\386.jpg'  
    image_dir = r'D:\Code\road_excavation_detect\data\test'
    output_image_dir = r'D:\Code\road_excavation_detect\output\super_resolution'  

    for image_name in os.listdir(image_dir):
        input_image_path = os.path.join(image_dir, image_name)
        upscale_image(input_image_path, output_image_dir, model='RealESRGAN_x2plus')
    # upscale_image(input_image_path, output_image_dir, model='RealESRGAN_x2plus')
    
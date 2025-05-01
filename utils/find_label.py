import os
import cv2
from ultralytics import YOLO


def find_label(image_path, output_dir, model):
    """
    Find labels in images using a YOLO model and save the results.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output images and crops.
        model: YOLO model for label detection.

    Returns:
        number_of_label: Number of labels detected in the image.
        labels_position: List of positions of the detected labels.
    """
    # Perform inference on the image
    results = model(image_path, save = True, project = output_dir, name='label', exist_ok=True, retina_masks=True)

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

        number_of_labels = len(xyxy)

        labels_position = xyxy.cpu().numpy().squeeze().astype(int)  # Convert to numpy array and squeeze

    # Save the results

    return number_of_labels, labels_position
    
    
if __name__ == '__main__':
    input_dir = './temp/crops'  # Directory containing cropped images
    output_dir = './temp/labels'  # Directory to save the output images and crops
    model =  YOLO('./model/find_label_best.pt') # Path to the YOLO model for label detection

    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)  # Full path to the image
        print(f"Processing image: {image_file}")
        nums, position= find_label(image_path=image_path, output_dir=output_dir, model=model)
        print(f"Image: {image_file}, Number of labels: {nums}, Positions: {position}")
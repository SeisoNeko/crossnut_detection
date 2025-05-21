import os
import cv2
import numpy as np
from cv2.typing import MatLike
from ultralytics import YOLO
if __name__ == "__main__":
    from hill_kmeans import hill_kmeans
else:
    from .hill_kmeans import hill_kmeans


def find_label(image: MatLike, img_name: str, output_dir: str, model: YOLO) -> tuple[int, np.ndarray, np.ndarray, list[MatLike]]:
    """
    Find labels in images using a YOLO model and save the results.

    Args:
        image (MatLike): Input image.
        img_name (str): File name of the input image.
        output_dir (str): Directory to save the output images.
        model: YOLO model for label detection.

    Returns:
        number_of_label: Number of labels detected in the image.
        labels_position: List of positions of the detected labels.
    """
    if not os.path.exists(os.path.join(output_dir, 'labels')):
        os.makedirs(os.path.join(output_dir, 'labels'))
    if not os.path.exists(os.path.join(output_dir, f'labels/{img_name}')):
        os.makedirs(os.path.join(output_dir, f'labels/{img_name}'))
        
    # if not os.path.exists(os.path.join(output_dir, 'kmeans')):
    #     os.makedirs(os.path.join(output_dir, 'kmeans'))    
    # image_kmeans = hill_kmeans(image)
    # cv2.imwrite(f"{output_dir}/kmeans/{img_name}_kmeans.png", image_kmeans)  # Save the k-means image
    
    ### Perform inference on the image
    results = model.predict(cv2.cvtColor(image, cv2.COLOR_BGRA2BGR), project=output_dir, name='labels', exist_ok=True, retina_masks=True, verbose=False, conf=0.3)
    # results = model.predict(image_kmeans, project=output_dir, name='labels', exist_ok=True, retina_masks=True, verbose=False, conf=0.3)
    
    
    result = results[0]  # one image only
    result.save(os.path.join(output_dir, f"labels/{img_name}/labels.png"))  # Save the results
    
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf.cpu().numpy().squeeze().astype(float) # confidence score of each box

    number_of_labels = len(xyxy)
    labels_position = xyxy.cpu().numpy().squeeze().astype(int)  # Convert to numpy array and squeeze

    labels = []  # Initialize an empty list to store the labels
    if number_of_labels > 0:
        labels = []
        labels_center = []
        # iterate each labels cordinates
        for i, (x1, y1, x2, y2) in enumerate(labels_position):
            crop = image[y1:y2, x1:x2]
            labels.append(crop)  # Crop the labels from the image
            labels_center.append(((x1+x2)//2, (y1+y2)//2))
            cv2.imwrite(f"{output_dir}/labels/{img_name}/label_{i}.png", crop)  # Save the cropped label image
            
        # Return the results
        labels_center = np.array(labels_center)
        return number_of_labels, labels_center, confs, labels
    
    return 0, np.array([]), np.array([]), []  # Return empty values if no labels found
    

    
    
if __name__ == '__main__':
    input_dir = './temp/crops'  # Directory containing cropped images
    output_dir = './temp'  # Directory to save the output images and crops
    model = YOLO('./model/find_label_best.pt') # Path to the YOLO model for label detection

    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)  # Full path to the image
        print(f"Processing image: {image_file}")
        nums, position, confs, labels = find_label(image_path=image_path, output_dir=output_dir, model=model)
        print(f"Image: {image_file}, Number of labels: {nums}\nPositions:\n{position}")
        print(f"Confs:\n{confs}")
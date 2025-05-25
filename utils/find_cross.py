import cv2
import numpy as np
import os
from cv2.typing import MatLike
from ultralytics import YOLO

def find_cross(image: MatLike, img_name: str, output_dir: str, model: YOLO) -> MatLike | None:
    """
    Find cross in images using a YOLO model and save the results.

    Args:
        image (Matlike): Input image.
        img_name (str): File name of the input image.
        output_dir (str): Directory to save the output images and crops.
        model: YOLO model for label detection.

    Returns:
        cross_image: cropped image with transparent background.
    """
    
    # check if the output path is valid
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'crops')):
        os.makedirs(os.path.join(output_dir, 'crops'))
    if not os.path.exists(os.path.join(output_dir, 'masked')):
        os.makedirs(os.path.join(output_dir, 'masked'))

    # results = model.predict(image_path, save_crop = True, project = output_dir, name='cross', exist_ok=True, retina_masks=True)
    results = model.predict(image, project = output_dir, name='cross', exist_ok=True, retina_masks=True, verbose=False)
    result = results[0] # one image only

    # can't find any cross in the image
    if len(result) == 0:
        print(f"No cross found in {img_name}.")
        return
    
    img = np.copy(result.orig_img)
    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # iterate each object contour
    for c in result:
        label = c.names[c.boxes.cls.tolist().pop()]
        b_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Create contour mask 
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        final_mask = cv2.bitwise_or(final_mask, b_mask)

    # Apply mask to image & Set alpha channel for transparency
    isolated = cv2.bitwise_and(img, img, mask=final_mask)
    isolated_gbra = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
    isolated_gbra[:, :, 3] = final_mask
    
    # mask (transparent background)
    cv2.imwrite(f"{output_dir}/masked/{img_name}_masked.png", isolated_gbra)
    
    # crop the image
    x1, y1, x2, y2 = result[0].boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)

    iso_crop = isolated[y1:y2, x1:x2]
    iso_crop_bgra = isolated_gbra[y1:y2, x1:x2]
    cv2.imwrite(f"{output_dir}/crops/{img_name}_{label}.png", iso_crop_bgra)
    
    return iso_crop_bgra


if __name__ == '__main__':
    cross_model = YOLO('./model/cross_best.pt')
    image_path = './pics/white.png'
    # image_path = './pics/colored_1/PXL_20250415_094322101.jpg'
    output_dir = './temp'
    
    find_cross(image_path=image_path, output_dir=output_dir, model=cross_model)
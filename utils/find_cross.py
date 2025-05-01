import cv2
import numpy as np
import os
from pathlib import Path

def find_cross(image_path, output_dir, model):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'crops'))  
    elif not os.path.exists(os.path.join(output_dir, 'crops')):
        os.makedirs(os.path.join(output_dir, 'crops'))
    if not os.path.exists(os.path.join(output_dir, 'masked')):
        os.makedirs(os.path.join(output_dir, 'masked'))
        

    results = model(image_path, save_crop = True, project = output_dir, name='color', exist_ok=True, retina_masks=True)
    
    result = results[0]
    img = np.copy(result.orig_img)
    img_name = Path(result.path).stem

    if len(result) == 0:
        pass
    
    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # iterate each object contour
    for ci,c in enumerate(result):
        label = c.names[c.boxes.cls.tolist().pop()]

        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask 
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        final_mask = cv2.bitwise_or(final_mask, b_mask)

    mask4ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGRA)
    mask4ch[:, :, 3] = final_mask # Set alpha channel to mask
    
    isolated = cv2.bitwise_and(img, img, mask=final_mask) # Apply mask to image
    isolated_rgba = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
    isolated_rgba[:, :, 3] = final_mask  # Set alpha channel for transparency
    
    # mask (弄黑)
    cv2.imwrite(f"{output_dir}/masked/{img_name}_masked.png", isolated_rgba)
    
    # 切照片  
    try:
        x1, y1, x2, y2 = result[0].boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
    except:
        print("No bounding box found in the image.")
        return

    iso_crop = isolated_rgba[y1:y2, x1:x2]
    cv2.imwrite(f"{output_dir}/crops/{img_name}_{label}.png", iso_crop)
            
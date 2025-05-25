import cv2
import numpy as np
import os
from cv2.typing import MatLike
from ultralytics import YOLO
if __name__ == "__main__":
    from find_cross_point import line_intersection
else:
    from .find_cross_point import line_intersection

# def get_edge_center(image: MatLike, h: int) -> list[int]:
#     row = image[h, :, 3]
#     edges = np.where((row[:-1] == 0) & (row[1:] == 255))[0] + 1
#     ends = np.where((row[:-1] == 255) & (row[1:] == 0))[0]
#     if row[0] == 255:
#         edges = np.insert(edges, 0, 0)
#     if row[-1] == 255:
#         ends = np.append(ends, len(row) - 1)
#     return ((edges + ends) // 2).tolist()

# def check_transparent(image: MatLike, x1: int, y1: int, x2: int, y2: int, tolerance_ratio: float = 0.03) -> bool:
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
#     alpha = image[:, :, 3]
#     line_alpha = cv2.bitwise_and(alpha, alpha, mask=mask)
    
#         # 統計：總像素數 / 不透明像素數
#     total_line_pixels = cv2.countNonZero(mask)
#     opaque_pixels = cv2.countNonZero(line_alpha)

#     # 容忍透明像素數量為圖寬 × ratio，向上取整避免精度誤差
#     max_transparent_pixels = int(np.ceil(min(image.shape[:2]) * tolerance_ratio))
#     transparent_pixels = total_line_pixels - opaque_pixels

#     return transparent_pixels <= max_transparent_pixels
    

def find_cross(image: MatLike, img_name: str, output_dir: str, model: YOLO) -> tuple[MatLike, tuple, tuple] | None:
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

    # crop_copy = np.copy(iso_crop_bgra)
    # tmp = []
    # height, width = crop_copy.shape[:2]

    # for n in range(2):
    #     bottom_candidates = []      
    #     first = True  
    #     for h1 in range(height // 3):
    #         done = False
    #         top_centers = get_edge_center(crop_copy, h1)
    #         if not top_centers: continue
                
    #         if first:
    #             first = False
    #             for h2 in range(height-1, height // 3, -1):
    #                 bottom_centers = get_edge_center(crop_copy, h2)
    #                 if bottom_centers:
    #                     bottom_candidates.append((h2, bottom_centers))

    #                 for x1 in top_centers:
    #                     for x2 in bottom_centers:
    #                         if check_transparent(crop_copy, x1, h1, x2, h2):
    #                             tmp.extend([(x1, h1), (x2, h2)] if n == 0 else [(h1, x1), (h2, x2)])
    #                             done = True
    #                             break
    #                     if done: break
    #                 if done: break
    #             if done: break
    #         else:
    #             for x1 in top_centers:
    #                 for h2, bottom_centers in bottom_candidates:
    #                     for x2 in bottom_centers:
    #                         if check_transparent(crop_copy, x1, h1, x2, h2):
    #                             tmp.extend([(x1, h1), (x2, h2)] if n == 0 else [(h1, x1), (h2, x2)])
    #                             done = True
    #                             break
    #                     if done: break
    #                 if done: break
    #             if done: break
        
    #     crop_copy = cv2.transpose(crop_copy)
    #     height, width = width, height
    
    # up, down, left, right = tmp
    # intersection = line_intersection((up, down), (left, right))
    
    
    # return iso_crop_bgra, intersection, (up, down)


if __name__ == '__main__':
    cross_model = YOLO('./model/cross_best.pt')
    image_path = './pics/white.png'
    # image_path = './pics/colored_1/PXL_20250415_094322101.jpg'
    output_dir = './temp'
    
    find_cross(image_path=image_path, output_dir=output_dir, model=cross_model)
            
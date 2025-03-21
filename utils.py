import os
import json

def convert_coco_to_yolo(json_path, output_path):
    json_path = os.path.join(json_path, '_annotations.coco.json')
    with open(json_path) as f:
        data = json.load(f)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image in data['images']:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']

        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        with open(os.path.join(output_path, f"{os.path.splitext(file_name)[0]}.txt"), 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height
                segmentation_points_list = []
                for segmentation in ann['segmentation']:
                    # Check if any element in segmentation is a string
                    if any(isinstance(point, str) for point in segmentation):
                        continue  # Skip this segmentation if it contains strings

                    segmentation_points = [str(min(max(float(point) / (width-1) if i % 2 == 0 else float(point) / (height-1), 0), 1)) for i, point in enumerate(segmentation)]
                    segmentation_points_list.append(' '.join(segmentation_points))
                    segmentation_points_string = ' '.join(segmentation_points_list)
                    line = '{} {}\n'.format(category_id, segmentation_points_string)
                    f.write(line)
                    segmentation_points_list.clear()
                    

                
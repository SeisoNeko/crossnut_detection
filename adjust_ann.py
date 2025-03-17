import json

# 讀取原始 JSON 檔案，假設檔名為 input.json
with open('_annotations.coco.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定義要保留的原始 category id
# keep_category_ids = {3, 4, 5}
keep_category_ids = {1}
# 過濾只保留那些標註其 category_id 在 keep_category_ids 內的 annotation
new_annotations = [ann for ann in data['annotations'] if ann['category_id'] in keep_category_ids]

# 定義新 category 的 id，這裡以 1 為例，並將所有保留的 annotation 的 category_id 修改為新 id
new_category_id = 0
for ann in new_annotations:
    ann['category_id'] = new_category_id

# 過濾只保留有標註的圖片（如果需要，也可以保留全部圖片）
annotated_image_ids = {ann['image_id'] for ann in new_annotations}
new_images = [img for img in data['images'] if img['id'] in annotated_image_ids]

# 建立新的 categories，統一為單一類別 "human"
new_categories = [{
    "id": new_category_id,
    "name": "cross",
    "supercategory": "none"  # 若需要，也可以設定成其他值或留空
}]

# 組合新的 COCO 資料結構
new_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": new_images,
    "annotations": new_annotations,
    "categories": new_categories
}

# 將處理後的結果存成 output.json
with open('_annotations.coco.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print("處理完成，新的 JSON 檔案已儲存為 output.json")

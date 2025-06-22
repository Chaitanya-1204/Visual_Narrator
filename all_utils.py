import json 
from collections import defaultdict



def build_json_data(file_path , output_file):
    
    with open(file_path , "r") as f:
        data = json.load(f)
    
    caption_dict = defaultdict(list)
    for ann in data["annotations"]:
        caption_dict[ann["image_id"]].append(ann["caption"])
        
    simplified_data = []
    for img in data["images"]:
        image_id = img["id"]
        captions = caption_dict.get(image_id, [])
        for cap in captions:
            entry = {
                "image_id": image_id,
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "caption": cap 
            }
            simplified_data.append(entry)
        
    with open(output_file, "w") as f:
        json.dump(simplified_data, f, indent=2)
    
    print("Saved data to " , output_file , " Total length : " , len(simplified_data))
    


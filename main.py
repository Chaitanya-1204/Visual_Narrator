import os 
from all_utils import build_json_data


train_file_path = "coco/annotations/captions_train2014.json"
val_file_path = "coco/annotations/captions_val2014.json"


train_output_path = "train_data.json"
val_output_path = "val_data.json"


if os.path.exists(train_output_path):
    print("Training Data already exists")

else:
    build_json_data(train_file_path , train_output_path)



if os.path.exists(val_output_path):
    print("Validation Data Exists")
else:
    build_json_data(val_file_path , val_output_path)
import json

json_files = [
    "Digest_ALL_crop_FFPE.json",
    "IHC_crop_new.json",
    "RJ_crop_lymphoma.json",
    "RUIJIN_crop.json",
    "TCGA_crop_FFPE.json",
    "TCGA_crop_Frozen.json",
    "Tsinghua_crop.json",
    "XIJING_crop.json"
]

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        print(file_path+":", len(json_data))
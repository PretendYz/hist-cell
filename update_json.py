import json
import os

root = "hist-cell/annotations"
categories = [
    {
        "supercategory": "1",
        "id": 0,
        "name": "1"
    },
    {
        "supercategory": "2",
        "id": 1,
        "name": "2"
    },
    {
        "supercategory": "3",
        "id": 2,
        "name": "3"
    },
    {
        "supercategory": "4",
        "id": 3,
        "name": "4"
    },
    {
        "supercategory": "5",
        "id": 4,
        "name": "5"
    }
]

# 打开文件，没有文件则自动新建，将字典写入文件中
for file in os.listdir(root):
    name = os.path.join(root, file)
    with open(name, "r") as f:
        data = json.load(f)
        data["categories"] = categories
    with open(name, "w") as f:
        json.dump(data, f)
#     关闭文件
# f.close()

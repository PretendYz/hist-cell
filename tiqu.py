import os
import shutil


# 获取所有部位的类别
def tiquImg():
    root = "hist-cell/gt"
    parts = ["Liver", "Esophagus", "Uterus", "Skin", "Testis", "Bile-duct",
             "Breast", "Pancreatic", "Cervix", "Lung", "Prostate", "Ovarian",
             "Adrenal", "HeadNeck", "Colon", "Kidney", "Stomach", "Bladder", "Thyroid"]
    for models in os.listdir(root):
        for part in parts:
            part_folder = os.path.join(os.path.join(root, models), part)
            if not os.path.exists(part_folder):
                os.mkdir(part_folder)
        for file in os.listdir(os.path.join(root, models)):
            if not file.endswith(".png"):
                continue
            part = file.split("_")[0]
            # print(part)
            old_path = os.path.join(os.path.join(root, models), file)
            new_path = os.path.join(os.path.join(os.path.join(root, models), part), file)
            shutil.copy(old_path, new_path)
            # print(old_path, new_path)


def tiqu(root, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    # 遍历所有子文件夹
    i = 0
    for folder, _, file in os.walk(root):
        if folder == dst:
            continue
        for f in file:
            old = os.path.join(folder, f)
            # print(old)
            shutil.copy(old, dst)
            # i = i + 1
    # print(i)
    #     for f in file:
    # if f.endswith(".json"):
    # print(f)
    # old = os.path.join(folder, f)
    # if folder.split("\\")[1] == "segm":
    #     continue
    # print(old, dst)
    # shutil.copy(old, dst)


if __name__ == '__main__':
    root = "hist-cell/result_img/pred"
    for f in os.listdir(root):
        src = os.path.join(root, f)
        dst = os.path.join(src, "All")
        # print(src, dst)
        tiqu(src, dst)
        # tiqu(root,)
    # dst = os.path.join(root, "All")
    # tiqu(root, dst)

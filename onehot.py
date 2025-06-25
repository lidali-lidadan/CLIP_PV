import numpy as np

train_list_path = 'VOCdevkit_PV/VOC2012/ImageSets/Segmentation/train_aug.txt'
save_path = 'VOCdevkit_PV/VOC2012/ImageSets/Segmentation/cls_labels_onehot.npy'

with open(train_list_path, 'r') as f:
    file_names = [line.strip() for line in f.readlines()]

label_dict = {}
for name in file_names:
    label_dict[name] = np.array([1.], dtype=np.float32)  # 只有光伏类，标签shape=(1,)

np.save(save_path, label_dict)
print(f"✅ 保存成功，标签 shape: {label_dict[file_names[0]].shape}")





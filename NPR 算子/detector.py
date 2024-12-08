import cv2
import torch
from torchvision.transforms.functional import to_pil_image
from networks.resnet import resnet50
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
import npr_option

model_path = './model_epoch_last_3090.pth'
# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
dir_name = './ProGAN/'
img_name = '00012.png'
img_origin = cv2.imread(dir_name + img_name)
img = npr_option.calculate_npr(img_origin, method='top')


# 使用 CAM 算法查看模型关注的区域
with SmoothGradCAMpp(model) as cam_extractor:
    # Preprocess your data and feed it to the model
    # 转换为 tensor，并且添加一个维度表示 batch 的数量，并且将通道数放在第二个维度
    _img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    _y_pred = model(_img).sigmoid().flatten()
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(_y_pred.squeeze(0).argmax().item(), _y_pred)
    # Visualize the raw CAM
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img_origin), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result);plt.axis('off');plt.tight_layout();plt.show()

model.train()
y_pred = model(_img).sigmoid().flatten().tolist()
y_pred_label = 'fake' if y_pred[0] > 0.5 else 'real'
print(y_pred_label)
print("Confidence: " + str(y_pred))

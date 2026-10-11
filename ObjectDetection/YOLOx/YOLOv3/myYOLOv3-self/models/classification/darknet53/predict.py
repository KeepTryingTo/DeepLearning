import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.resnet_sp import resnet50
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 5
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet50(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = r"weights/0.92_best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # load image
    root = r'images'
    for img_name in os.listdir(root):
        img_path = os.path.join(root,img_name)
        img = Image.open(img_path)

        img_t = data_transform(img)
        # expand batch dimension
        img_t = torch.unsqueeze(img_t, dim=0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img_t.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "truth class: {}  predict class: {}   prob: {:.3}".format(
            img_name.split('.')[0],
            class_indict[str(predict_cla)],
            predict[predict_cla].numpy()
        )

        print(f'======={img_name.split(".")[0]} prediction result==========')
        print(print_res)
        plt.imshow(img)
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        plt.show()
        print('============================================================')


if __name__ == '__main__':
    main()

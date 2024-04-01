from __future__ import print_function, division
import torch
from torchvision import  transforms

# style 분류
def classify_by_style(img):
    torch.multiprocessing.freeze_support()
   
    #저장된 모델 불러오기
    # checkpoint = torch.load('./k_fashion/checkpoint/kfashion_style/model_best.pth.tar', map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    model = torch.jit.load('./k_fashion/model/9oz_style_model.pt', map_location='cpu')
    model.eval()
    
    # 이미지 변형
    transform_image = transforms.Compose([
    #   transforms.ToPILImage(), already an image
      transforms.Resize((224,224)),
      transforms.ToTensor()
      # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformed_image = transform_image(img).unsqueeze(0)

    # 모델에 전달
    output = model(transformed_image)
    _,pred = torch.max(output,1)
    style_result=pred.cpu().detach().numpy()[0]
    # print("style",pred.cpu().detach().numpy()[0])

    return style_result

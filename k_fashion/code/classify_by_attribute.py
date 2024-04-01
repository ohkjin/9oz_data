from k_fashion.utility.resnest import *
from k_fashion.utility.helpers import *
from k_fashion.utility.util import *

import argparse
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim



parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--image-size', default=224, type=int)
parser.add_argument('-j', '--workers', default=12, type=int)
parser.add_argument('--device_ids', default=[0,1,2,3], type=int, nargs='+')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                    help='evaluate model on validation set')

def run_attribute_classifier(model_name ='category',img=None):
    torch.multiprocessing.freeze_support()
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    state = {'batch_size': args.batch_size, 'image_size': args.image_size}
    state['use_gpu'] = use_gpu
    state['workers'] = args.workers
    state['device_ids'] = args.device_ids

    if model_name == 'category':
        num_classes = 21
        state['resume'] = './k_fashion/checkpoint/kfashion_category/model_category_best.pth.tar'
    elif model_name == 'detail':
        num_classes = 40
        state['resume'] = './k_fashion/checkpoint/kfashion_detail/model_detail_best.pth.tar'
    elif model_name == 'texture':
        num_classes = 27
        state['resume'] = './k_fashion/checkpoint/kfashion_texture/model_texture_best.pth.tar'
    elif model_name == 'print':
        num_classes = 21
        state['resume'] = './k_fashion/checkpoint/kfashion_print/model_print_best.pth.tar'
    
    model = resnest50d(pretrained=False, nc=num_classes)
    load_checkpoint(model,state['resume'],use_gpu)

    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
    transform_image = transforms.Compose([
        Warp(state['image_size']),
        transforms.ToTensor(),
        normalize,
    ])
    transformed_image = transform_image(img)
    tensor_list = [transformed_image, transformed_image]
    # Stack the tensors along a new dimension to create a single tensor
    stacked_tensor = torch.stack(tensor_list)
    # [2,3,224,224]
    if use_gpu:
            # oz_loader.pin_memory = True
            cudnn.benchmark = True
            # num_devices = torch.cuda.device_count()
            model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            # criterion = criterion.cuda()
    
    # feature = torch.unsqueeze(transformed_image, 0) # ([3, 224, 224])=>([1, 3, 224, 224])
    feature_var = torch.autograd.Variable(stacked_tensor).float()
    # print("feature_var",feature_var.shape)
    # print("feature", stacked_tensor.shape)
    with torch.no_grad():
    # compute output
        output = model(stacked_tensor)
    # print("output", output.shape, output)
    output_cpu = output[0].cpu()
    # print("output_cpu",output_cpu)
    # best_n = np.argsort(output_cpu, axis=1)[:,-1:] # tensor ([[3],[3],[2]...])
    best_n = torch.argmax(output_cpu).item()
    # print("best",best_n)
    return best_n



# if __name__ == '__main__':
#     startTime = time.time()
#     print("Starting..")
def classify_by_attribute(img):
    lst = []
    model_names = ['category','detail','texture','print']
    for att in model_names:
        p1 = int(run_attribute_classifier(model_name=att,img=img))
        # 숫자를 문자로 치환
        if att == 'category':
            mapping_dict = {0: 'JK', 1: 'PT', 2: 'JP', 3: 'SK', 4: 'CA', 5: 'JP', 6: 'TS',
                            7: 'WS', 8: 'PT', 9: 'OP', 10: 'CT', 11: 'DP', 12: 'ST', 13: 'KT',
                            14: 'VT', 15: 'CT', 16: 'TN', 17: 'BL', 18: 'TN', 19: 'TS', 20: 'LG'}
        if att == 'detail':
            mapping_dict = {0: '스터드', 1: '드롭숄더', 2: '드롭웨이스트', 3: '레이스업', 4: '슬릿', 5: '프릴', 6: '단추', 7: '퀄팅', 8: '스팽글',
                            9: '롤업', 10: '니트꽈베기', 11: '체인', 12: '프린지', 13: '지퍼', 14: '태슬', 15: '띠', 16: '플레어', 17: '싱글브레스티드',
                            18: '더블브레스티드', 19: '스트링', 20: '자수', 21: '폼폼', 22: '디스트로이드', 23: '페플럼', 24: 'X스트랩', 25: '스티치',
                            26: '레이스', 27: '퍼프', 28: '비즈', 29: '컷아웃', 30: '버클', 31: '포켓', 32: '러플', 33: '글리터', 34: '퍼트리밍',
                            35: '플리츠', 36: '비대칭', 37: '셔링', 38: '패치워크', 39: '리본'}
        if att == 'texture':
            mapping_dict = {0: "패딩", 1: "무스탕", 2: "퍼프", 3: "네오프렌", 4: "코듀로이", 5: "트위드", 6: "자카드", 7: "니트", 8: "페플럼",
                            9: "레이스", 10: "스판덱스", 11: "메시", 12: "비닐/PVC", 13: "데님", 14: "울/캐시미어", 15: "저지", 16: "시퀸/글리터",
                            17: "퍼", 18: "헤어 니트", 19: "실크", 20: "린넨", 21: "플리스", 22: "시폰", 23: "스웨이드", 24: "가죽", 25: "우븐",
                            26: "벨벳"}
        if att == 'print':
            mapping_dict = {0: "페이즐리", 1: "하트", 2: "지그재그", 3: "깅엄", 4: "하운즈 투스", 5: "도트", 6: "레터링", 7: "믹스", 8: "뱀피",
                            9: "해골", 10: "체크", 11: "무지", 12: "카무플라쥬", 13: "그라데이션", 14: "스트라이프", 15: "호피", 16: "아가일",
                            17: "그래픽", 18: "지브라", 19: "타이다이", 20: "플로럴"}
        n1 = mapping_dict[p1]
        lst.append(n1)
    return lst

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
from PIL import Image
from image_segmentation.show_segment import show_segment
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np

def image_segment(img):
    clothes_classification={
        'upper':0.00,
        'skirt':0.00,
        'pants':0.00,
        'dress':0.00,
    }
    #-- image segmentation --#
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    pred_array = pred_seg.numpy().tolist()
    # print("pred_seg_upper", int((pred_seg==4).sum()))
    # print("pred_seg_np_upper", int((pred_seg.numpy()==4).sum()))
    # print("pred_seg_arr_upper", any(4 in sublist for sublist in pred_array))
    # print("pred_seg",pred_seg.numpy().tolist().size)
    # plt_img = show_segment(model,pred_seg)
    # plt.imshow(pred_seg)
    # plt.show()
    #-- plt to image --#
    # img_buf = io.BytesIO()
    # plt.savefig(img_buf, format='png')
    # plt_img = Image.open(img_buf)

    #-- image percentage --#
    upper_pixels = int((pred_seg==4).sum())
    skirt_pixels = int((pred_seg==5).sum())
    pants_pixels = int((pred_seg==6).sum())
    dress_pixels = int((pred_seg==7).sum())
    total_pixels = upper_pixels + skirt_pixels + pants_pixels + dress_pixels

    clothes_classification['upper']=round(upper_pixels/total_pixels*100,1)
    clothes_classification['skirt']=round(skirt_pixels/total_pixels*100,1)
    clothes_classification['pants']=round(pants_pixels/total_pixels*100,1)
    clothes_classification['dress']=round(dress_pixels/total_pixels*100,1)

    #-- image masking --#
    # background = Image.new('RGBA', img.size, color=(0, 0, 0, 0)) # 4 차원
    background = Image.new('RGB', img.size, color=(0, 0, 0))
    print("img.size",list(img.size))
    bbox_array = [[],[],[],[],[img.size[0],img.size[1]]]
    masked_cropped_array = [None,None,None,None]
    cate = ['upper','skirt','pants','dress']

    for i in range(4):
        if(clothes_classification[cate[i]]<=15.0):
            continue
        binary_mask = (pred_seg==i+4).detach().numpy()
        masked = Image.composite(img, background, Image.fromarray(binary_mask))
        # Find the bounding box of the segmented object
        nonzero_pixels = np.argwhere(binary_mask)  # Get the coordinates of nonzero (True) elements
        if len(nonzero_pixels) > 0:
            # Calculate the bounding box
            bbox = (np.min(nonzero_pixels[:, 1]),           # Minimum x-coordinate (left)
                    np.min(nonzero_pixels[:, 0]),           # Minimum y-coordinate (top)
                    np.max(nonzero_pixels[:, 1]),  # Width
                    np.max(nonzero_pixels[:, 0]))  # Height
            print("Bounding Box (x, y, w, h):", bbox)
        # else:
            # print("No object found in segmentation mask.")
        masked_cropped = masked.crop(bbox)
        masked_cropped_array[i] = masked_cropped
        # print("list(bbox)",list(bbox))
        bbox_array[i] = [int(item) for item in bbox]
    # print("bbox_array",bbox_array)
    # print("masked_cropped_array",masked_cropped_array)
    return clothes_classification, masked_cropped_array, bbox_array
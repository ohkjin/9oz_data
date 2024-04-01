import io
import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from PIL import Image
import torch
import base64
from image_segmentation.image_segment import image_segment
from k_fashion.code.classify_by_style import classify_by_style
from k_fashion.code.classify_by_attribute import classify_by_attribute
from resnet_vae.code.resnet_vae import ResNet_VAE
from resnet_vae.code.cosine_similarity import get_similar_images


app = Flask(__name__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/')
def hello():
    return 'Hello'

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Test by POSTMAN by file upload
@app.route('/test', methods=['POST'])
def test():   
    #-- image receive --#
    request_obj = request.files['file']
    # print('request_obj',request_obj)
    image_bytes = io.BytesIO(request_obj.read())
    img = Image.open(image_bytes)
    #-- data to send --#
    response_obj={
        'predSeg':[],
        'upper':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'skirt':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'pants':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'dress':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        }
    }
    #-- 이미지 전송 --#
    # img_base64 = base64.b64encode(image_bytes.read())
    # print(type(img))
    # files = {'file': open(img, 'rb')}
    # response = requests.post('http://springboot_server:port/receive_image', files=files)
    # return jsonify({'file':str(img_base64)})
    # return send_file(image_file, mimetype='image/jpeg')
    
    #-- image segmentation (dict)--#
    classification, masked_img_array, bbox_array = image_segment(img)
    # print("pred_seg",int((pred_seg==4)))
    response_obj['predSeg']= bbox_array
    response_obj['upper']['percentage'] = classification['upper']
    response_obj['skirt']['percentage'] = classification['skirt']
    response_obj['pants']['percentage'] = classification['pants']
    response_obj['dress']['percentage'] = classification['dress']
    # dress_masked.show()
    # dress_masked.save(os.path.join('uploads', 'dress_masked.png'))

    #-- ResNetVae similar images --#
    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
    CNN_embed_dim = 256     # latent dim extracted by 2D CNN
    res_size = 224        # ResNet image size
    dropout_p = 0.2       # dropout probability
    resnet_vae = ResNet_VAE(resnet_model=34, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim)
    resnet_vae.load_state_dict(torch.load('./resnet_vae/model/34/model_epoch45.pth'))
    # print("Using", torch.cuda.device_count(), "GPU!")
    model_params = list(resnet_vae.parameters())
    pickle_name = './resnet_vae/model/34/product_resnet_vae_34_encoded_epoch45_240326.pkl'
    with open('product_names.pkl', 'rb') as f:
        product_names = pickle.load(f)    
    product_amount = 13956
    top=5
    #-- style/attribute classification (style dict) & ResNetVae & --#
    percentage_limit = 20.0

    cate = ['upper','skirt','pants','dress']
    similarities = [[],[],[],[]]
    encoded_images = [None,None,None,None]
    # print(masked_img_array[0]==masked_img_array[3])
    for i in range(4):
        if masked_img_array[i]==None:
            continue
        # masked_img_array[i].show()
        # print("masked_photo_mode",masked_photo.mode)
        response_obj[cate[i]]['style'] = int(classify_by_style(masked_img_array[i]))
        # Dictionary mapping seasons to corresponding textures
        
        attribute_lst = classify_by_attribute(masked_img_array[i])
        response_obj[cate[i]]['kindId'] = attribute_lst[0]
        response_obj[cate[i]]['detail'] = attribute_lst[1]
        response_obj[cate[i]]['texture'] = attribute_lst[2]
        response_obj[cate[i]]['print'] = attribute_lst[3]
        # print("attribute_lst",attribute_lst)
        seasons_mapping = {
            '봄': ["트위드", "코듀로이", "시폰", "벨벳", "저지", "스판덱스", "페플럼" "우븐", "데님" "니트","네오프렌","자카드","레이스"],
            '여름': ["린넨", "실크", "시퀸/글리터", "메시", "비닐/PVC"],
            '겨울': ["패딩", "무스탕", "퍼프", "울/캐시미어", "퍼", "가죽", "스웨이드", "플리스", "헤어 니트"]
        }
        for key, val in seasons_mapping.items():
            if val == response_obj[cate[i]]['texture']:
                response_obj[cate[i]]['season'] = key
        
        #ResNetVae
        top_indices, similarities[i], encoded_images[i]= get_similar_images(masked_img_array[i], resnet_vae, pickle_name, product_amount, top)
        # print(f"{cate[i]} top", top_indices)
        response_obj[cate[i]]['similar'] = [product_names[i] for i in top_indices]
        
    # print("similarities equal", np.array_equal(similarities[0],similarities[3]))
    # print("encoded images",len(encoded_images),torch.equal(encoded_images[0],encoded_images[3]))
    return jsonify(response_obj)



@app.route('/photo_to_flask', methods=['POST'])
def getPhotoInput():
    # BE에서 json객체 전달받기
    request_obj = request.get_json()
    # print('request_obj',request_obj)
    # 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress"
    #-- data to send --#
    response_obj={
        'predSeg':[[]],
        'upper':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'skirt':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'pants':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'dress':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        }
    }
    if request_obj['image']!='':
   
        # 이미지 받기
        byte_file = request_obj['image'] ## byte file
        base64_file = base64.b64decode(byte_file)
        img = Image.open(io.BytesIO(base64_file)).convert("RGB")
        # img.show()
        # print("size",img.size)
        # print("mode",img.mode)

        #-- image segmentation (dict)--#
        classification, masked_img_array, bbox_array = image_segment(img)
        response_obj['predSeg']= bbox_array
        response_obj['upper']['percentage'] = classification['upper']
        response_obj['skirt']['percentage'] = classification['skirt']
        response_obj['pants']['percentage'] = classification['pants']
        response_obj['dress']['percentage'] = classification['dress']
        # dress_masked.show()

        #-- ResNetVae similar images --#
        # EncoderCNN architecture
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
        CNN_embed_dim = 256     # latent dim extracted by 2D CNN
        res_size = 224        # ResNet image size
        dropout_p = 0.2       # dropout probability
        resnet_vae = ResNet_VAE(resnet_model=34, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim)
        resnet_vae.load_state_dict(torch.load('./resnet_vae/model/34/model_epoch45.pth'))
        # print("Using", torch.cuda.device_count(), "GPU!")
        model_params = list(resnet_vae.parameters())
        pickle_name = './resnet_vae/model/34/product_resnet_vae_34_encoded_epoch45_240326.pkl'
        with open('product_names.pkl', 'rb') as f:
            product_names = pickle.load(f)    
        product_amount = 13956
        top=100
        #-- style classification (style dict) & ResNetVae --#
        percentage_limit = 20.0
        cate = ['upper','skirt','pants','dress']
        similarities = [[],[],[],[]]
        encoded_images = [None,None,None,None]
        for i in range(4):
            if masked_img_array[i]==None:
                continue
            # masked_img_array[i].show()
            # print("masked_photo_mode",masked_photo.mode)
            response_obj[cate[i]]['style'] = int(classify_by_style(masked_img_array[i]))
            # Dictionary mapping seasons to corresponding textures
            
            attribute_lst = classify_by_attribute(masked_img_array[i])
            response_obj[cate[i]]['kindId'] = attribute_lst[0]
            response_obj[cate[i]]['detail'] = attribute_lst[1]
            response_obj[cate[i]]['texture'] = attribute_lst[2]
            response_obj[cate[i]]['print'] = attribute_lst[3]
            # print("attribute_lst",attribute_lst)
            seasons_mapping = {
                '봄': ["트위드", "코듀로이", "시폰", "벨벳", "저지", "스판덱스", "페플럼" "우븐", "데님" "니트","네오프렌","자카드","레이스"],
                '여름': ["린넨", "실크", "시퀸/글리터", "메시", "비닐/PVC"],
                '겨울': ["패딩", "무스탕", "퍼프", "울/캐시미어", "퍼", "가죽", "스웨이드", "플리스", "헤어 니트"]
            }
            for key, val in seasons_mapping.items():
                if val == response_obj[cate[i]]['texture']:
                    response_obj[cate[i]]['season'] = key
            
            #ResNetVae
            top_indices, similarities[i], encoded_images[i]= get_similar_images(masked_img_array[i], resnet_vae, pickle_name, product_amount, top)
            response_obj[cate[i]]['similar'] = [product_names[i] for i in top_indices]
            
        # print("similarities equal", np.array_equal(similarities[0],similarities[3]))
        # print("encoded images",len(encoded_images),torch.equal(encoded_images[0],encoded_images[3]))
        # print("response_obj",response_obj)
        return jsonify(response_obj)
    
@app.route('/photo_to_segment', methods=['POST'])
def getPhotoSegment():
    # BE에서 json객체 전달받기
    request_obj = request.get_json()
    # print('request_obj',request_obj)
    # 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress"
    #-- data to send --#
    response_obj={
        'predSeg':[[]],
        'upper':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'skirt':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'pants':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        },
        'dress':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'kindId':'CD',
            'detail':'',
            'texture':'',
            'print':'',
            'similar':[],
        }
    }
    if request_obj['image']!='':
   
        # 이미지 받기
        byte_file = request_obj['image'] ## byte file
        base64_file = base64.b64decode(byte_file)
        img = Image.open(io.BytesIO(base64_file)).convert("RGB")
        # img.show()
        # print("size",img.size)
        # print("mode",img.mode)

        #-- image segmentation (dict)--#
        classification, masked_img_array, bbox_array = image_segment(img)
        response_obj['predSeg']= bbox_array
        response_obj['upper']['percentage'] = classification['upper']
        response_obj['skirt']['percentage'] = classification['skirt']
        response_obj['pants']['percentage'] = classification['pants']
        response_obj['dress']['percentage'] = classification['dress']
        # dress_masked.show()

        #-- style classification (style dict) & ResNetVae --#
        percentage_limit = 20.0
        cate = ['upper','skirt','pants','dress']
        similarities = [[],[],[],[]]
        encoded_images = [None,None,None,None]
        for i in range(4):
            if masked_img_array[i]==None:
                continue
            # masked_img_array[i].show()
            # print("masked_photo_mode",masked_photo.mode)
            response_obj[cate[i]]['style'] = int(classify_by_style(masked_img_array[i]))
            # Dictionary mapping seasons to corresponding textures
            attribute_lst = classify_by_attribute(masked_img_array[i])
            response_obj[cate[i]]['kindId'] = attribute_lst[0]
            response_obj[cate[i]]['detail'] = attribute_lst[1]
            response_obj[cate[i]]['texture'] = attribute_lst[2]
            response_obj[cate[i]]['print'] = attribute_lst[3]
            # print("attribute_lst",attribute_lst)
            seasons_mapping = {
                '봄': ["트위드", "코듀로이", "시폰", "벨벳", "저지", "스판덱스", "페플럼" "우븐", "데님" "니트","네오프렌","자카드","레이스"],
                '여름': ["린넨", "실크", "시퀸/글리터", "메시", "비닐/PVC"],
                '겨울': ["패딩", "무스탕", "퍼프", "울/캐시미어", "퍼", "가죽", "스웨이드", "플리스", "헤어 니트"]
            }
            for key, val in seasons_mapping.items():
                if val == response_obj[cate[i]]['texture']:
                    response_obj[cate[i]]['season'] = key
            
        return jsonify(response_obj)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

#코사인 유사도
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle



def get_similar_images(img, model, pickle_name, product_amount, top):
    # img.show()
    transform_image = transforms.Compose([
    #   transforms.ToPILImage(), already an image
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ])
    transformed_image = transform_image(img)

    # Add two same tensor since single tensor doesn't get encoded
    # Create a list containing both tensors
    tensor_list = [transformed_image, transformed_image]
    # Stack the tensors along a new dimension to create a single tensor
    stacked_tensor = torch.stack(tensor_list)
    # [2,3,224,224]
    with torch.no_grad():
        mus,logvars = model.encode(stacked_tensor)
    # print("model",model)
    mu = mus[0]
    logvar = logvars[0]
    encoded_image= model.reparameterize(mu,logvar)
    # print("encoded_image",encoded_image)
    print("pickle",pickle_name)
    with open(pickle_name, 'rb') as f:
        product_encoded = pickle.load(f)

    similarities = cosine_similarity(encoded_image.reshape(1, -1), product_encoded.reshape(product_amount, -1))
    print("similarities",similarities[0])

    # Get indices of the top 5 similar images
    top_indices = np.argsort(similarities[0])[-top:][::-1]

    # Visualize the randomly selected image and the top 5 similar images
    # plt.figure(figsize=(10, 2))
    # plt.subplot(1, 6, 1)
    # plt.imshow(random_image.permute(1, 2, 0).numpy())
    # plt.title(f'Random Image\nLabel: {random_label}')
    # plt.axis('off')


    # for i, idx in enumerate(top5_indices):
    #     similar_image, similar_label = custom_dataset[idx]
    #     plt.subplot(1, 6, i + 2)
    #     plt.imshow(similar_image.permute(1, 2, 0).numpy())
    #     plt.title(f'Similar {i + 1}\nLabel: {similar_label}\nSimilarity: {similarities[0, idx]:.4f}')
    #     plt.axis('off')

    # plt.show()
    return top_indices, similarities[0],encoded_image
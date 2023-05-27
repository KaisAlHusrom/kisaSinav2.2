import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import concurrent.futures
import numpy as np

# Load pre-trained ResNet18 model
model = resnet18(weights="IMAGENET1K_V1")
model.eval()


# girilen resimlerini modelin gereksinimler ile uyusmak icin donusturmek.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Bir görüntüden özelliklerini çıkarma fonksiyonu

def extract_features(image):
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # Add batch dimension
    features = model(image)
    features = torch.flatten(features)  # Flatten the features
    return features.detach().numpy()


# Kosinüs benzerliğini kullanarak iki görüntü arasındaki benzerliği hesaplama
def compute_similarity(image_pair):
    image_path1, image_path2 = image_pair
    # debugging için görüntü yollarını yazdırma
    print("Processing:", image_path1, image_path2)

    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")

    features1 = extract_features(image1)
    features2 = extract_features(image2)

    similarity = 1 - torch.nn.functional.cosine_similarity(
        torch.from_numpy(features1), torch.from_numpy(features2), dim=0
    )

    return similarity.item()


if __name__ == '__main__':
    # resimlerin yerini tutan liste
    image_paths = []
    with open("images/names.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            image_paths.append(f"images/{line.rstrip()}")

    # resimlerin karşılaştırması ve benzerliğini kaydetmek
    similarity_matrix = {}
    image_pairs = [(image_paths[i], image_paths[j]) for i in range(
        len(image_paths)) for j in range(i + 1, len(image_paths))]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        similarities = executor.map(compute_similarity, image_pairs)

    for image_pair, similarity in zip(image_pairs, similarities):
        similarity_matrix[image_pair] = similarity

    print(similarity_matrix)

    # Benzerlik eşiğini ayarlama
    threshold = 0.01

    # Eşiğe göre benzer resimler bulma
    similar_images = [(image_pair[0], image_pair[1], similarity) for image_pair,
                      similarity in similarity_matrix.items() if similarity > threshold]

    print(similar_images)

    # Benzer görüntü çiftlerini yazdırma
    for image_pair in similar_images:
        print(
            f"Similar images: {image_pair[0]} and {image_pair[1]}, similarity: {round(image_pair[2], 6)}")

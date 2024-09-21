import vectorization
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def vectors_data(data_path):
    vectors = vectorization.feature_extract()
    model = vectors.get_model_extract()
    vectors.store_vector(model, data_path)

def search_img(img_path):
    image = vectorization.feature_extract()
    model = image.get_model_extract()
    img_search_vector = image.vector_normalized(model, img_path)

    with open("vectors.pkl", "rb") as f:
        vectors = pickle.load(f)
    with open("paths.pkl", "rb") as f:
        paths = pickle.load(f)

    distance = np.linalg.norm(vectors - img_search_vector, axis=1)

    ids = np.argsort(distance)[:10]
    nearest_image = [(paths[id], distance[id]) for id in ids]

    return nearest_image


if __name__ == "__main__":
    #vectors_data('dataset')
    nearest_image = search_img('testimg/cheetah1.jpg')
    axes = []
    grid_size = int(math.sqrt(10))
    fig = plt.figure(figsize=(10,5))


    for id in range(10-1):
        draw_image = nearest_image[id]
        axes.append(fig.add_subplot(grid_size, grid_size, id+1))

        axes[-1].set_title(draw_image[1])
        plt.imshow(Image.open(draw_image[0]))

    fig.tight_layout()
    plt.show()



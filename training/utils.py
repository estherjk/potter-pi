import math
import matplotlib.pyplot as plt
import numpy as np

def show_batch(image_batch, label_batch):
    """
    Show a single batch of images.
    """

    grid_size = math.ceil(math.sqrt(BATCH_SIZE))

    plt.figure(figsize=(5,5))
    for n in range(BATCH_SIZE):
        ax = plt.subplot(grid_size, grid_size, n+1)

        # Reshape image array from (x,y,1) to (x,y)
        img = np.squeeze(image_batch[n])

        plt.imshow(img, cmap='gray')
        plt.title('Label: ' + str(label_batch[n]), fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_test_batch(image_batch, label_batch, predictions):
    """
    Show the predicted results.
    """

    grid_size = math.ceil(math.sqrt(len(predictions)))

    plt.figure(figsize=(10,10))
    for n in range(len(predictions)):
        ax = plt.subplot(grid_size, grid_size, n+1)

        # Reshape image array from (x,y,1) to (x,y)
        img = np.squeeze(image_batch[n])

        plt.imshow(img, cmap='gray')
        plt.title('Label: ' + str(label_batch[n]), fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Pred: ' + str(np.around(predictions[n], decimals=2)), fontsize=8)

    plt.tight_layout()
    plt.show()
import numpy as np
import torchvision
import torchvision.transforms as transforms

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        npad = ((4, 4), (4, 4), (0, 0))
        image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
        # print(image.shape)
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        upper_left_i = np.random.randint(9)
        upper_left_j = np.random.randint(9)

        image = image[upper_left_i : upper_left_i+32, upper_left_j : upper_left_j + 32, :]
        # print(image.shape)
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        flip_bit = np.random.randint(2)

        if(flip_bit == 1):
            image = np.fliplr(image)

        # print(image.shape)
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image,axis=(0,1), keepdims=True)
    std = np.mean(image,axis=(0,1), keepdims=True)
    # print(mean.shape, image.shape)
    image = (image - mean)/std
    # print(np.mean(image))
    # print(np.var(image), np.std(image), np.mean(image))
    ### YOUR CODE HERE

    return image
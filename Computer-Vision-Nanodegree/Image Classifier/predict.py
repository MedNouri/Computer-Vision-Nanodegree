import argparse
import json

import numpy as np
import torch
import torchvision
from PIL import Image


def load_checkpoint(filepath):
    """
    Load a saved checkpoint and rebuild the model.

    Args:
        filepath: File path of the saved checkpoint.

    Returns:
        model: Rebuilt model with loaded checkpoint parameters.
    """
    # Load the checkpoint
    checkpoint = torch.load(filepath)

    # Load the appropriate pre-trained VGG model based on the checkpoint type
    if checkpoint['vgg_type'] == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load the classifier and state_dict from the checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model, and returns a Numpy array.

    Args:
        image_path: File path of the image to be processed.

    Returns:
        Tensor: Processed image as a PyTorch tensor.
    """
    # Open the image using PIL
    pil_image = Image.open(image_path)

    # Resize the image
    pil_image = pil_image.resize((256, 256))

    # Center crop the image
    width, height = pil_image.size
    new_width, new_height = 224, 224
    left = round((width - new_width) / 2)
    top = round((height - new_height) / 2)
    right = left + new_width
    bottom = top + new_height
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert color channel from 0-255 to 0-1
    np_image = np.array(pil_image) / 255

    # Normalize the image for the model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose color channels to the first dimension
    np_image = np_image.transpose((2, 0, 1))

    # Convert the processed image to a Float Tensor
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)

    return tensor


def predict(image_path, model, topk, device, cat_to_name):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    Args:
        image_path: File path of the image to be predicted.
        model: Trained deep learning model for prediction.
        topk: Number of top classes to return.
        device: Device to use for prediction (cpu or cuda).
        cat_to_name: Mapping of category indices to real names.

    Returns:
        Tuple containing the list of probabilities for top classes and the list of predicted flower names.
    """
    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # Move to the specified device
    image = image.to(device)
    model.eval()

    # Perform prediction
    with torch.no_grad():
        ps = torch.exp(model(image))

    # Get topK classes and their probabilities
    ps, top_classes = ps.topk(topk, dim=1)

    # Map indices to flower names
    idx_to_flower = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]

    # Return the probabilities and predicted flower names as lists
    return ps.tolist()[0], predicted_flowers_list


def print_predictions(args):
    """
    Load the model checkpoint
    Args:
        args: Parsed arguments containing image_filepath, model_filepath, category_names_json_filepath, top_k, and gpu.

    Returns:
        None
    """
    # Load the model checkpoint
    model = load_checkpoint(args.model_filepath)

    # Determine the device based on user arguments and device availability
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not torch.cuda.is_available():
        device = 'cpu'
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'

    model = model.to(device)

    # Load the category names mapping from the JSON file
    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # Predict the image classes
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    # Print the predictions
    print("Predictions:")
    for i in range(args.top_k):
        print("#{: <3} {: <25} Prob: {:.2f}%".format(i, top_classes[i], top_ps[i] * 100))


if __name__ == '__main__':
    # Create the argument parser and add arguments
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(dest='image_filepath', help="File path of the image you want to classify")
    parser.add_argument(dest='model_filepath', help="File path of the checkpoint file, including the extension")

    # Optional arguments
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath',
                        help="File path to a JSON file that maps categories to real names",
                        default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k',
                        help="Number of most likely classes to return, default is 5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu',
                        help="Include this argument if you want to use the GPU for processing",
                        action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Print the predictions
    print_predictions(args)

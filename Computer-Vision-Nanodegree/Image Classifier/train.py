import argparse
import os

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data


def data_transformation(args):
    """
    Transform the training and validation data, create DataLoader objects, and return class_to_idx dictionary.

    Args:
        args: Parsed arguments containing data_directory and save_directory.

    Returns:
        train_data_loader: DataLoader for training data.
        valid_data_loader: DataLoader for validation data.
        class_to_idx: Dictionary mapping class labels to indices.
    """

    # Define train and valid directories
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    # Validate paths before proceeding
    if not os.path.exists(args.data_directory):
        raise FileNotFoundError("Data Directory doesn't exist: {}".format(args.data_directory))
    if not os.path.exists(args.save_directory):
        raise FileNotFoundError("Save Directory doesn't exist: {}".format(args.save_directory))
    if not os.path.exists(train_dir):
        raise FileNotFoundError("Train folder doesn't exist: {}".format(train_dir))
    if not os.path.exists(valid_dir):
        raise FileNotFoundError("Valid folder doesn't exist: {}".format(valid_dir))

    # Define transformations for training and validation data
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create ImageFolder datasets for training and validation
    train_data = ImageFolder(root=train_dir, transform=train_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)

    # Create DataLoader objects for training and validation data
    train_data_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data_loader = data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return train_data_loader, valid_data_loader, train_data.class_to_idx


def build_model(args):
    """
    Build the model using the specified pre-trained architecture and return the model with the updated classifier.

    Args:
        args: Parsed arguments containing model_arch.

    Returns:
        model: Pre-trained model with the updated classifier.
    """
    if args.model_arch == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif args.model_arch == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif args.model_arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif args.model_arch == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features_of_pretrained_model = model.classifier[0].in_features

    classifier = nn.Sequential(
        nn.Linear(in_features=in_features_of_pretrained_model, out_features=2048, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=2048, out_features=102, bias=True),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    return model


def train_model(args, model, train_data_loader, valid_data_loader, class_to_idx):
    """
    Train the model, save it to the specified directory, and return True if successful.

    Args:
        args: Parsed arguments containing learning_rate, epochs, gpu.
        model: Pre-trained model with the updated classifier.
        train_data_loader: DataLoader for training data.
        valid_data_loader: DataLoader for validation data.
        class_to_idx: Dictionary mapping class labels to indices.

    Returns:
        True if the model training and saving process is successful.
    """
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)

    print_every = 20

    for e in range(args.epochs):
        step = 0
        running_train_loss = 0

        for images, labels in train_data_loader:
            step += 1
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

            if step % print_every == 0 or step == 1 or step == len(train_data_loader):
                print("Epoch: {}/{} Batch % Complete: {:.2f}%".format(e + 1, args.epochs,
                                                                      (step) * 100 / len(train_data_loader)))

        model.eval()
        with torch.no_grad():
            running_accuracy = 0
            running_valid_loss = 0
            for images, labels in valid_data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
                running_valid_loss += valid_loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            average_train_loss = running_train_loss / len(train_data_loader)
            average_valid_loss = running_valid_loss / len(valid_data_loader)
            accuracy = running_accuracy / len(valid_data_loader)
            print("Train Loss: {:.3f}".format(average_train_loss))
            print("Valid Loss: {:.3f}".format(average_valid_loss))
            print("Accuracy: {:.3f}%".format(accuracy * 100))

    model.class_to_idx = class_to_idx
    checkpoint = {
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'epochs': args.epochs,
        'optim_stat_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'vgg_type': args.model_arch
    }

    torch.save(checkpoint, os.path.join(args.save_directory, "checkpoint.pth"))
    print("Model saved to {}".format(os.path.join(args.save_directory, "checkpoint.pth")))
    return True


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(dest='data_directory',
                        help="Directory containing training images. Expect 'train' and 'valid' folders inside.")

    # Optional arguments with improved help messages
    parser.add_argument('--save_directory', dest='save_directory',
                        help="Directory to save the trained model. Default is '../saved_models'.")
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help="Learning rate for model training. Default is 0.003. Expected type: float.",
                        default=0.003, type=float)
    parser.add_argument('--epochs', dest='epochs',
                        help="Number of epochs for model training. Default is 3. Expected type: int.",
                        default=3, type=int)
    parser.add_argument('--gpu', dest='gpu',
                        help="Use GPU for training if available.", action='store_true')
    parser.add_argument('--model_arch', dest='model_arch',
                        help="Pre-trained model architecture to use. Default is 'vgg19'. Choose from: vgg11, vgg13, vgg16, vgg19.",
                        default="vgg19", type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])

    # Parse the arguments
    args = parser.parse_args()

    # Load and transform the data
    train_data_loader, valid_data_loader, class_to_idx = data_transformation(args)

    # Train the model and save it
    train_model(args, build_model(args), train_data_loader, valid_data_loader, class_to_idx)

import os
import matplotlib.pyplot as plt
from detecto import core, utils
from torchvision import transforms
from dataset import create_dir
import configuration as config


train_xml_folder = config.train_xml_folder
val_xml_folder = config.test_xml_folder
train_image_folder = config.images_path
val_image_folder = config.images_path
test_image_folder = config.images_path
labels = config.labels
epochs = config.epochs
lr = config.lr
gamma = config.gamma
lr_step_size = config.lr_step_size
batch_size = config.batch_size
###################################################################################################
TRAIN_LABELS_CSV = f'{config.csv_path}/train.csv'
VAL_LABELS_CSV = f'{config.csv_path}/test.csv'
model_folder = config.model_folder
model_file = config.model_file
saved_model_file_path = os.path.join(model_folder, model_file)

loss_graph = config.loss_graph
graph_path = config.loss_path
saved_graph_file_path = os.path.join(graph_path, loss_graph)


def train():
    # Define custom transforms to apply to your dataset
    custom_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(800),
        transforms.ColorJitter(saturation=0.3),
        transforms.ToTensor(),
        utils.normalize_transform(),
    ])

    # Pass in a CSV file instead of XML files for faster Dataset initialization speeds
    train_dataset = core.Dataset(TRAIN_LABELS_CSV, train_image_folder, transform=custom_transforms)

    # Create our validation dataset
    val_dataset = core.Dataset(VAL_LABELS_CSV, val_image_folder)  # Validation dataset for training

    # Create your own DataLoader with custom options
    loader = core.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = core.Model(labels)
    losses = model.fit(loader, val_dataset, epochs=epochs, learning_rate=lr, gamma=gamma,
                       lr_step_size=lr_step_size, verbose=True)

    # plt.plot(losses)  # Visualize loss throughout training
    # plt.show()
    model.save(saved_model_file_path)  # Save model to a file


if __name__ == "__main__":
    train()

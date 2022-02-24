from detecto import utils, core
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import matplotlib.colors as mcolors

import configuration as config

colours = list(mcolors.TABLEAU_COLORS.keys())
label_color = dict(zip(config.labels, colours))
model = core.Model.load('Trained_Models/model_weights.pth', ['Signature'])


def save_objects_snippet(identified_objects, image_file):
    image = cv2.imread(image_file)
    filename_without_extension = image_file.split('.')[0]
    filename_without_extension = filename_without_extension.split('\\')[-1]
    for k, v in identified_objects.items():
        for i, coord in enumerate(v):
            cropped_img = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
            cv2.imwrite(f'predicted_output/{filename_without_extension}_{i + 1}.jpg', cropped_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100,
                         cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            plt.imshow(cropped_img)
            plt.show()


def visualize_predicted_objects(image, preds, score_filter, title=""):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.imshow(image)
    identified_objects = {}
    for label, box, score in zip(*preds):
        if score >= score_filter:
            if label not in identified_objects.keys():
                identified_objects[label] = []
            identified_objects[label].append(box.tolist())
            width, height = box[2] - box[0], box[3] - box[1]
            initial_pos = (box[0], box[1])
            rect = patches.Rectangle(initial_pos, width, height, linewidth=2, edgecolor=label_color[label],
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0] + 2, box[1] - 2, f'{label}: {round(score.item(), 2)}', color=label_color[label])
        ax.set_title(f'{title}')
    plt.show()
    return identified_objects


def test_single_file(model, image_file, score_filter, title=""):
    image = utils.read_image(image_file)
    preds = model.predict(image)
    # If already a tensor, reverse normalize it and turn it back
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(utils.reverse_normalize(image))

    identified_objects = visualize_predicted_objects(image, preds, score_filter, title)
    save_objects_snippet(identified_objects, image_file)


if __name__ == '__main__':
    test_single_file(model, 'MicrosoftTeams-image.png', 0.40)

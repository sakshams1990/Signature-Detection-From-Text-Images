import os
import random
import shutil

from detecto import utils
import configuration as config

image_folder = config.images_path
train_xml_dir = config.train_xml_folder
test_xml_dir = config.test_xml_folder
csv_dir = config.csv_path


# create directory if not exists
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# split xml files into train and test
def split_images_into_train_test(xml_path, split_ratio):
    create_dir(train_xml_dir)  # Create train xml folder
    create_dir(test_xml_dir)  # Create test xml folder
    no_of_xml_files = len(os.listdir(xml_path))  # number of xml files in the folder
    list_of_xml_files = os.listdir(xml_path)  # List of xml files in the folder
    no_of_train_images = int(split_ratio * no_of_xml_files)  # number of xml files for training
    train_xml_files = random.sample(list_of_xml_files, no_of_train_images)  # randomly selected xml files
    for file in list_of_xml_files:
        if file in train_xml_files:
            srcpath = os.path.join(xml_path, file)
            shutil.copy(srcpath, train_xml_dir)  # Copy files from Images to train folder
        else:
            srcpath = os.path.join(xml_path, file)
            shutil.copy(srcpath, test_xml_dir)  # Copy files from Images to test folder


# Create a csv file from xml files
def csv_from_xml(xml_path, csv_file):
    create_dir(csv_dir)
    utils.xml_to_csv(xml_path, f'{config.csv_path}/{csv_file}.csv')


def main():
    split_images_into_train_test(config.annotation_path, config.split_ratio)
    csv_from_xml(train_xml_dir, 'train')
    csv_from_xml(test_xml_dir, 'test')


if __name__ == '__main__':
    main()

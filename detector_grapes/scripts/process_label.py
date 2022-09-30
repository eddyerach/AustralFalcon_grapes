# Python code used to format VGG labels JSON file into detectron2 compatible JSON files for training.
# This code separates the JSON from VGG into individual JSON for each image that contains labels, matches each image in the dataset that has been 
# label with its corresponding JSON, and separates into "train" and "test" folders in a ratio of 80% and 20 % respectively
# Example of how to run the code in terminal:
# process_label.py --via_file path/to/VGG_labels_file.json --images_src path/to/labeled_dataset_images --images_dst path/to/dataset_folder_train_test
# type process_label.py --h or process_label.py --help for details

import json
import os 
import argparse
import glob
import shutil 
import random

global parser

parser = argparse.ArgumentParser(description="Detectron 2 script formater. Takes 1 viaproject json file as input")
parser.add_argument(
        "--via_file",
        metavar="FILE",
        help="Path to via file",
    )
parser.add_argument(
        '--images_src',
        help='Labeled dataset images path'
)

parser.add_argument(
        '--images_dst',
        help='Dest of coppied images (test and train folders will be created)'
)
args = parser.parse_args()


def get_individual_files(via_file):
    
    print('On get_individual_files')
    with open(via_file) as json_file:
        data = json.load(json_file)
        dict_instances = {}
        for key, value in data.items():
            if value["regions"]:
                dict_instances[key] = value
    return dict_instances

def generate_files(dict_instances):
    print('On generate_files')
    for key, value in dict_instances.items():
        dict_instance = {}
        dict_instance['version'] = '4.2.9'
        dict_instance['flags'] = {}
        dict_instance['shapes'] = []
        for idx, region in enumerate(value['regions']):
            dict_instance_sub = {}
            dict_instance_sub['label'] = 'disease'
            points = list(map(list,list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))))
            dict_instance_sub['points'] = points
            dict_instance_sub['group_id'] = None
            dict_instance_sub['shape_type'] = 'polygon'
            dict_instance['shapes'].append(dict_instance_sub)
        name = key.split('.')[0] 
        dict_instance['imagePath'] = name + '.jpg'
        dict_instance['imageData'] = 'test'
        dict_instance['imageHeight'] = 600
        dict_instance['imageWidth'] = 800
        
        with open(name+'.json', 'w') as fp:
            json.dump(dict_instance, fp)

def split_dataset(train_portion, json_files_list, images_src, images_dst):

    random.seed(4)
    random.shuffle(json_files_list)

    train_data = json_files_list[:round(len(json_files_list) * train_portion)]
    test_data = json_files_list[round(len(json_files_list) * train_portion):]

    print(f'Num train: {len(train_data)} Num test: {len(test_data)}')

    for train_file in train_data:
        image = train_file.rstrip().replace("json", "jpg")
        label = train_file 
        image_src_path = os.path.join(images_src,image)
        image_dst_path = os.path.join(images_dst,image)
        image_dst_path_train = os.path.join(images_dst,'train',image)
        label_path = label
        label_dst_path = os.path.join(images_dst,label)
        label_dst_path_train = os.path.join(images_dst,'train',label)
       
        try:
            shutil.copy(image_src_path, image_dst_path)
        except:
            print(f'Train 1File: {image_src_path} cannot be moved into {image_dst_path}')
        try:
            shutil.copy(image_src_path, image_dst_path_train)
        except:
            print(f'Train 2File: {image_src_path} cannot be moved int {image_dst_path_train}')
        #Copy label into general folder and also train folder
        try:
            shutil.copy(label_path, label_dst_path)
            shutil.copy(label_path, label_dst_path_train)
        except:
            print(f'File: {label_path} cannot be moved')
    
    for test_file in test_data:
        image = test_file.rstrip().replace("json", "jpg")
        label = test_file 
        image_src_path = os.path.join(images_src,image)
        image_dst_path = os.path.join(images_dst,image)
        image_dst_path_test = os.path.join(images_dst,'test',image)
        label_path = label
        label_dst_path = os.path.join(images_dst,label)
        label_dst_path_test = os.path.join(images_dst,'test',label)
        
        #Copy image into general folder and also train folder        
        try:
            shutil.copy(image_src_path, image_dst_path)
            shutil.copy(image_src_path, image_dst_path_test)
        except:
            print(f'File: {image_src_path} cannot be moved')
        #Copy label into general folder and also train folder
        try:
            shutil.copy(label_path, label_dst_path)
            shutil.copy(label_path, label_dst_path_test)
        except:
            print(f'File: {label_path} cannot be moved')


        

if __name__ == "__main__":
    print('ON Main')
    
    #Get data from images with the presence of symptoms 
    print(f'via_file: {args.via_file}')
    dict_individual_files = get_individual_files(args.via_file)

    #Generate json label file for each image 
    generate_files(dict_individual_files)

    #Get list of all json files present (Excluding the global one)
    json_files_list = glob.glob('*.json')

    #Split labels in train and test dataset
    train_portion = 0.8

    split_dataset(train_portion, json_files_list, args.images_src, args.images_dst)
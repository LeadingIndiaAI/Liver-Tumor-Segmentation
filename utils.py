import os

def file_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            return dirs


def save_file2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    sub_dirs = file_name_path(file_dir)
    out.writelines("filename" + "\n")
    for index in range(len(sub_dirs)):
        out.writelines(file_dir + "/" + sub_dirs[index] + "\n")


save_file2csv("/storage/research/Intern19_v2/segmentation/Team 1/TrainingData/", "train_X.csv")
save_file2csv("/storage/research/Intern19_v2/segmentation/Team 1/TrainingDataMasks/", "train_Y.csv")

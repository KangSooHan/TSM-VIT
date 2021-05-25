import os

PATH_DATASET = '/media/HDD3'

def return_ucf101(modality):
    filename_categories = 'UCF101/file_list/category.txt'
    if modality == 'RGB':
        root_data = 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/train_videofolder.txt'
        filename_imglist_val = 'UCF101/file_list/val_videofolder.txt'
        prefix = 'frame{:06d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv1(modality):
    filename_categories = 'STST_v1/category.txt'
    if modality == 'RGB':
        root_data = 'STST_v1/20bn-something-something-v1'
        filename_imglist_train = 'STST_v1/train_videofolder.txt'
        filename_imglist_val = 'STST_v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/file_list/20bn-something-something-v1-flow'
        filename_imglist_train = 'UCF101/file_list/train_videofolder_flow.txt'
        filename_imglist_val = 'UCF101/file_list/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



def return_dataset(dataset, modality):
    dict_single = {'something_v1' : return_somethingv1, "something_v2" : return_somethingv2,
                    'ucf101' : return_ucf101, 'hmdb51' : return_hmdb51, 'kinetics' : return_kinetics }

    if dataset in dict_single:
        file_categories, imglist_train, imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError("Unknown Dataset " + dataset)

    imglist_train = os.path.join(PATH_DATASET, imglist_train)
    imglist_val = os.path.join(PATH_DATASET, imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(PATH_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()

        categories = [item.rstrip() for item in lines]

    else:
        categories = [None] * file_categories

    n_class = len(categories)
    print("{} : {} classes".format(dataset, n_class))
    return n_class, imglist_train, imglist_val, root_data, prefix

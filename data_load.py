def return_dataset(dataset, modality):
    dict_single = {'something_v1' : return_something, "something_v2" : return_something,
                    'ucf101' : return_ucf101, 'hmdb51' : return_hmdb51, 'kinetics' : return_kinetics }

    if dataset in dict_single:
        file_categories, imglist_train, imglist_val, root_data, prefix = dict_single[dataset](modality)

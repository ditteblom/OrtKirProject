import sys
sys.settrace()
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from utils import train, predict, score_fluoroscopy, score_time, score_xray, score_retries_cannulated_dhs, score_retries_hansson, \
                drill_dist_hansson, guidewire_dist, drill_dist_hansson
from skimage.io import imread_collection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimulationData(torch.utils.data.Dataset):
    def __init__(self, repair_type, split, transform = None, data_path = "Data", train_size = 0.8, test_size = 0.2, seed = 8):
        'Initializing data'
        self.data_path = data_path
        self.repair_type = repair_type + "/001_copenahgen_test_1"
        #self.type = type
        self.transform = transform

        img_files = glob.glob(self.data_path + '***/**/*.jpg', recursive=True)
        img_files = [ x for x in img_files if self.repair_type in x ]
        score_files = glob.glob(self.data_path + '***/**/*.txt', recursive=True)
        score_files = [ x for x in score_files if self.repair_type in x ]
        scores = []
        scores_true = []
        maxscore = 0
                
        for file in score_files:
            with open(file) as f:
                lines = f.read()
            idx_score_end = lines.find('Score')
            tmp = lines[:idx_score_end]
            idx_score_start = tmp.rfind('\n')
            score = np.double(lines[idx_score_start+1:idx_score_end-4])

            if maxscore == 0:
              idx_maxscore_end = lines.find('Max score')
              tmp = lines[:idx_maxscore_end]
              idx_maxscore_start = tmp.rfind('\n')
              maxscore = np.double(lines[idx_maxscore_start+1:idx_maxscore_end-3])

            scores_true.append(score/maxscore)

            # find variables to be corrected in score
            if repair_type == "001_hansson_pin_system":
              var_score = []
              variables = ["Fluoroscopy", "Total time", "Nr of X-rays", "Nr of retries", "Distal drill distance to joint surface (mm)",
                          "Guide wire distance to joint surface (mm)", "Proximal drill distance to joint surface (mm)"]
              for var in variables:
                idx_end = lines.find(var)
                tmp = lines[:idx_end]
                idx_start = tmp.rfind('\n')
                var_score.append(np.double(lines[idx_start+1:idx_end-4]))

              score += score_fluoroscopy(var_score[0])
              score += score_time(var_score[1])
              score += score_xray(var_score[2])
              score += score_retries_hansson(var_score[3])
              score += drill_dist_hansson(var_score[4])
              score += guidewire_dist(var_score[5])
              score += drill_dist_hansson(var_score[6])

              if score > maxscore:
                score = maxscore

            scores.append(score/maxscore)

        # create dataframe with filenames for frontal images
        df_frontal = pd.DataFrame(img_files, columns = ["image_path_frontal"])
        df_frontal = df_frontal[df_frontal.image_path_frontal.str.contains('|'.join(["frontal"]))==True]
        df_frontal["no"] = df_frontal.image_path_frontal.apply(lambda x: x[-19:-4]) # get the unique ending of the filename

        # create dataframe with filenames for lateral images
        df_lateral = pd.DataFrame(img_files, columns = ["image_path_lateral"])
        df_lateral = df_lateral[df_lateral.image_path_lateral.str.contains('|'.join(["lateral"]))==True]
        df_lateral["no"] = df_lateral.image_path_lateral.apply(lambda x: x[-19:-4]) # get the unique ending of the filename

        # create dataframe with scores
        df_scores = pd.DataFrame(score_files, columns = ["no"])
        df_scores["true scores"] = scores_true
        df_scores["score"] = scores
        df_scores.no = df_scores.no.apply(lambda x: x[-19:-4])

        # merge the three dataframes
        df = df_frontal.merge(df_lateral, how = 'left', on = 'no')
        df = df.merge(df_scores, how = 'left', on = 'no')
        df = df[df.image_path_frontal.str.contains('|'.join(["admin","guest","resultTableImage"]))==False]
        df = df[df.image_path_lateral.str.contains('|'.join(["admin","guest","resultTableImage"]))==False]\
                .loc[~(df["true scores"]<=0)] # remove all admin and guest files and the images of the results. Remove all black
                                    # black images which have a score of 0
        #if type is not None:
        #    df = df[df.image_path.str.contains('|'.join([type]))==True]
        #df.reset_index(drop=True, inplace = True)

        df.to_csv("data.csv")

        # get path for frontal images
        #df_frontal = df[df.image_path.str.contains('|'.join(["frontal"]))==True]
        frontal_paths = df.image_path_frontal.tolist() #[df.image_path.str.contains('|'.join(["frontal"]))==True].tolist()

        # get path for lateral images
        #df_lateral = df[df.image_path.str.contains('|'.join(["lateral"]))==True]
        lateral_paths = df.image_path_lateral.tolist()

        image_paths = []
        for i in range(len(frontal_paths)):
          image_paths.append([frontal_paths[i],lateral_paths[i]]) # stack frontal and lateral paths

        #image_paths = df.image_path.tolist()
        scores_list = df.score.tolist()

        image_trainval, image_test, score_trainval, score_test = train_test_split(image_paths, scores_list, test_size=test_size, train_size=train_size, random_state=seed)
        image_train, image_val, score_train, score_val = train_test_split(image_trainval, score_trainval, test_size=0.2, train_size=0.8, random_state=seed)

        # devide images into train, validation and test set
        if split == "train":
            self.images, self.scores = image_train, score_train
        elif split == "val":
            self.images, self.scores = image_val, score_val
        elif split == "test":
            self.images, self.scores = image_test, score_test
        else:
          print("Please provide either train, val or test as split.")

        
    def __len__(self):
        'Returns the number of samples'
        return len(self.images)

    def __getitem__(self,idx):
        'Generate one sample of data'
        image_path = self.images[idx]
        images = imread_collection([image_path[0], image_path[1]], conserve_memory=True) # stack the frontal and lateral images
        score = self.scores[idx]
        if self.transform:
            #image = transforms.functional.invert(image) # convert image to negative (like Xray images)
            #image = self.transform(image) # perform transforms
            images = torch.stack([self.transform(img) for img in images], dim = 1) # transform both frontal and lateral images
            #images = images.permute(images, (1,0,2,3))
        return images, score

def get_loader(repair_type, split, data_path = "Data", batch_size=16, transform = None, num_workers=0):
    """Build and return a data loader."""
    
    dataset = SimulationData(repair_type, split, data_path, transform, train_size = 0.8, test_size = 0.2, seed = 8)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
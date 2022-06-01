from __future__ import annotations
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import PIL.Image as Image
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from utils import train, predict, score_fluoroscopy, score_time, score_xray, score_retries_cannulated_dhs, score_retries_hansson, \
                drill_dist_hansson, guidewire_dist, drill_dist_hansson, drill_dhs, stepreamer_dist, drill_dist_cannulated, guidesize_cannulated
from skimage.io import imread_collection
import cv2
import pandas as pd
import numpy as np

experts = ['HenrikV',
'hpalm',
'jalv003',
'LarsL',
'MadsV',
'NanaS',
'PeterS',
'PeterT',
'ThomasB']

def if_expert(row):
    for expert in experts:
        if expert in row:
            return 1
        else:
            return 0

eps = 10e-16

class SimulationData(torch.utils.data.Dataset):
    def __init__(self, repair_type, split, data_path = "/work3/dgro/Data/", transform = None, train_size = 0.8, test_size = 0.2, seed = 8):
        'Initializing data'
        self.data_path = data_path
        self.repair_type = repair_type + "/001_copenahgen_test_1"
        self.transform = transform

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

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
              variables = ["Fluoroscopy (normalized)", "Total time", "Nr of X-rays", "Nr of retries", "Distal drill distance to joint surface (mm)",
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
            
            if repair_type == "029_dynamic_hip_screw":
              var_score = []
              variables = ["Fluoroscopy (normalized)", "Time", "Nr of X-rays", "Nr of retries", "3.2 mm drill outside cortex (mm)",
                          "Guide wire distance to joint surface (mm)", "Step reamer distance to joint surface (mm)"]
              for var in variables:
                idx_end = lines.find(var)
                if idx_end == -1 and var == "Nr of X-rays":
                    var = 'Number of X-rays'
                    idx_end = lines.find(var)
                elif idx_end == -1 and var == "Nr of retries":
                    var = 'Number of retries'
                    idx_end = lines.find(var)
                tmp = lines[:idx_end]
                idx_start = tmp.rfind('\n')
                var_score.append(np.double(lines[idx_start+1:idx_end-4]))

              score += score_fluoroscopy(var_score[0])
              score += score_time(var_score[1])
              score += score_xray(var_score[2])
              score += score_retries_cannulated_dhs(var_score[3])
              score += drill_dhs(var_score[4])
              score += guidewire_dist(var_score[5])
              score += stepreamer_dist(var_score[6])

              if score > maxscore:
                score = maxscore

            if repair_type == "028_cannulated_screws":
              var_score = []
              variables = ["Fluoroscopy (normalized)", "Time", "Number of X-Rays", "Nr of retries",
                             "Inferior guide wire distance to joint surface","Posterior guide wire distance to joint surface",
                             "Inferior drill distance to joint surface","Posterior drill distance to joint surface",
                             "Guide size"
                             ]
              for var in variables:
                idx_end = lines.find(var)
                tmp = lines[:idx_end]
                idx_start = tmp.rfind('\n')
                var_score.append(np.double(lines[idx_start+1:idx_end-4]))

              score += score_fluoroscopy(var_score[0])
              score += score_time(var_score[1])
              score += score_xray(var_score[2])
              score += score_retries_cannulated_dhs(var_score[3])
              score += guidewire_dist(var_score[4])
              score += guidewire_dist(var_score[5])
              score += drill_dist_cannulated(var_score[6])
              score += drill_dist_cannulated(var_score[7])
              score += guidesize_cannulated(var_score[8])

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

        # load assessment on train data
        assess_train_path = repair_type + '_traindata_randomized_AG.csv'
        assess_train = pd.read_csv(assess_train_path, index_col = 0, delimiter=',')
        assess_train = assess_train[['no','assessment']]

        # merge with data
        data = df.merge(assess_train, how = 'left', on = 'no') # now -> data + train

        # load assessment on test data
        assess_path = repair_type + '_testdata_randomized_AG.csv'
        assess = pd.read_csv(assess_path, index_col = 0, delimiter=',')
        assess = assess[['no','assessment']]

        # merge with data
        data = data.merge(assess, how = 'left', on = 'no') # now -> data + train + test

        # assessment will be in two columns now -> combine
        data = data.fillna(0)
        data['assessment'] = data.apply(lambda x: x['assessment_x'] + x['assessment_y'],axis=1)
        data = data.drop(['assessment_x','assessment_y'], axis = 1)
        data['assessment'] = data.assessment.replace(0, np.nan)

        # assign expert status
        data['expert']= data.image_path_frontal.apply(if_expert)

        data.to_csv(repair_type + "_data.csv")

        # get path for frontal images
        frontal_paths = data.image_path_frontal.tolist() #[df.image_path.str.contains('|'.join(["frontal"]))==True].tolist()

        # get path for lateral images
        lateral_paths = data.image_path_lateral.tolist()

        image_paths = []
        for i in range(len(frontal_paths)):
          image_paths.append([frontal_paths[i],lateral_paths[i]]) # stack frontal and lateral paths

        # convert to lists
        scores_list = data.score.tolist()
        scores_list = torch.Tensor(scores_list).to(self.device)

        assess_list = data.assessment.tolist()
        assess_list = torch.Tensor(assess_list).to(self.device)

        expert_list = data.expert.tolist()
        expert_list = torch.Tensor(expert_list).to(self.device)

        image_trainval, image_test, score_trainval, score_test, assess_trainval, assess_test = train_test_split(image_paths, scores_list, assess_list, test_size=test_size, train_size=train_size, random_state=seed)
        image_train, image_val, score_train, score_val, assess_train, assess_val = train_test_split(image_trainval, score_trainval, assess_trainval, test_size=0.2, train_size=0.8, random_state=seed)

        # save train and test as csv files
        image_test_df = pd.DataFrame([image_test,score_test, assess_test])
        image_test_df.to_csv(repair_type +'_testdata.csv')
        image_train_df = pd.DataFrame([image_train,score_train,assess_train])
        image_train_df.to_csv(repair_type +'_traindata.csv')
        
        # devide images into train, validation and test set
        if split == "train":
            self.images, self.scores, self.assess = image_train, score_train, assess_train
        elif split == "val":
            self.images, self.scores, self.assess = image_val, score_val, assess_val
        elif split == "test":
            self.images, self.scores, self.assess = image_test, score_test, assess_test
        else:
            print("Please provide either train, val or test as split.")

        
    def __len__(self):
        'Returns the number of samples'
        return len(self.images)

    def __getitem__(self,idx):
        'Generate one sample of data'
        # get path from init
        image_path = self.images[idx]
        # read in images
        images = imread_collection([image_path[0], image_path[1]], conserve_memory=True) # stack the frontal and lateral images
        # resize to 900x900 and crop
        transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((900,900)),
                                    ])
        images_resized = [transform(img) for img in images]
        images_crop = []
        # the indices for the cropping are qualitatively assessed.
        # Should be modified to fit the images for the specific problem.
        images_crop += [np.array(images_resized[0])[50:650,200:800]/255.0]
        images_crop += [np.array(images_resized[1])[100:800,100:800]/255.0]

        if self.transform:
            images = torch.stack([self.transform(img) for img in images_crop], dim = 1) # transform both frontal and lateral images

        # get score and assessment for the images
        score = self.scores[idx]
        assess = self.assess[idx]
        return images.float(), score.float(), assess.float()

def get_loader(repair_type, split, data_path = "/work3/dgro/Data/", batch_size=16, transform = None, num_workers=0, shuffle = True, seed = 8):
    """Build and return a data loader."""
    
    dataset = SimulationData(repair_type, split, data_path, transform, train_size = 0.8, test_size = 0.2, seed = seed)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)

    print('Finished loading dataset.')
    return data_loader
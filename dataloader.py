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
                drill_dist_hansson, guidewire_dist, drill_dist_hansson
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

        assess = pd.read_csv('data_randomized_AG_020522.csv', index_col = 0, delimiter=';')
        assess = assess[['no','assessment']]

        data = df.merge(assess, how = 'left')
        data['expert']= data.image_path_frontal.apply(if_expert)

        data.to_csv("data.csv")

        # get path for frontal images
        #df_frontal = df[df.image_path.str.contains('|'.join(["frontal"]))==True]
        frontal_paths = data.image_path_frontal.tolist() #[df.image_path.str.contains('|'.join(["frontal"]))==True].tolist()

        # get path for lateral images
        #df_lateral = df[df.image_path.str.contains('|'.join(["lateral"]))==True]
        lateral_paths = data.image_path_lateral.tolist()

        image_paths = []
        for i in range(len(frontal_paths)):
          image_paths.append([frontal_paths[i],lateral_paths[i]]) # stack frontal and lateral paths

        #image_paths = df.image_path.tolist()
        scores_list = data.score.tolist()
        scores_list = torch.Tensor(scores_list).to(self.device)

        assess_list = data.assessment.tolist()
        assess_list = torch.Tensor(assess_list).to(self.device)

        expert_list = data.expert.tolist()
        expert_list = torch.Tensor(expert_list).to(self.device)

        image_trainval, image_test, score_trainval, score_test, assess_trainval, assess_test = train_test_split(image_paths, scores_list, assess_list, test_size=test_size, train_size=train_size, random_state=seed)
        image_train, image_val, score_train, score_val, assess_train, assess_val = train_test_split(image_trainval, score_trainval, assess_trainval, test_size=0.2, train_size=0.8, random_state=seed)

        self.mean_ = 'log(1-scores)' #torch.mean(score_train)
        self.std_ = 'log(1-scores)' #torch.mean(score_trainval)
        
        # devide images into train, validation and test set
        if split == "train":
            self.images, self.scores, self.assess = image_train, score_train, assess_train#torch.log(1-score_train+eps)#(score_train-self.mean_)/self.std_
        elif split == "val":
            self.images, self.scores, self.assess = image_val, score_val, assess_val#torch.log(1-score_val+eps)#(score_val-self.mean_)/self.std_
        elif split == "test":
            self.images, self.scores, self.assess = image_test, score_test, assess_test#torch.log(1-score_test+eps)#(score_test-self.mean_)/self.std_
        else:
            print("Please provide either train, val or test as split.")

        
    def __len__(self):
        'Returns the number of samples'
        return len(self.images)

    def __getitem__(self,idx):
        'Generate one sample of data'
        image_path = self.images[idx]
        # # load plain image (without screws)
        # plainFrontal = cv2.imread('/work3/dgro/frontal.jpg')
        # plainFrontal = cv2.cvtColor(plainFrontal, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # plainLateral = cv2.imread('/work3/dgro/lateral.jpg')
        # plainLateral = cv2.cvtColor(plainLateral, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # # load image with screws
        # grayFrontal = cv2.imread(image_path[0])
        # grayFrontal = cv2.cvtColor(grayFrontal, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # grayLateral = cv2.imread(image_path[1])
        # grayLateral = cv2.cvtColor(grayLateral, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # #### Frontal ####
        # # initialize SIFT
        # sift = cv2.SIFT_create()
        # # find keypoints and descriptors in each image
        # keypoints_plain, descriptors_plain = sift.detectAndCompute(plainFrontal,None)
        # keypoints_im, descriptors_im = sift.detectAndCompute(grayFrontal,None)
        # # initialize brute force matcher
        # bf = cv2.BFMatcher(crossCheck=True)
        # # Match descriptors.
        # matches = bf.match(descriptors_plain,descriptors_im)
        # # find source and distance points
        # src_pts = np.float32([ keypoints_plain[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        # dst_pts = np.float32([ keypoints_im[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        # # Find Homography matrix
        # M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # # find shape of plain (without screws) image
        # rows, cols = plainFrontal.shape
        # anFrontal = cv2.warpPerspective(plainFrontal,M,(rows,cols))#[:image.shape[0],:image.shape[1]]
        # if anFrontal.shape[0] > grayFrontal.shape[0]:
        #     anFrontal = anFrontal[:grayFrontal.shape[0],:]
        # else:
        #     grayFrontal = grayFrontal[:anFrontal.shape[0],:]
        # if anFrontal.shape[1] > grayFrontal.shape[1]:
        #     anFrontal = anFrontal[:,:grayFrontal.shape[1]]
        # else:
        #     grayFrontal = grayFrontal[:,:anFrontal.shape[1]]
        # assert anFrontal.shape == grayFrontal.shape, "Annotation and image should be same size"
        # # subtract image to extract screws
        # anFrontal = (anFrontal-grayFrontal)
        # # threshold image
        # anFrontal = np.where(anFrontal>29, 1, 0).astype(np.uint8)
        # # # crop out background and pad to fit original shape
        # crop_size_x, crop_size_y = int(np.floor((anFrontal.shape[0]*(3/4)))), int(np.floor((anFrontal.shape[1]*(3/4))))
        # pad_x, pad_y = int((anFrontal.shape[0]-(crop_size_x-1))/2), int((anFrontal.shape[1]-(crop_size_y-1))/2)
        # anFrontal = transforms.CenterCrop((crop_size_x,crop_size_y))(torch.tensor(anFrontal))
        # anFrontal = (transforms.Pad((pad_x,pad_y))(anFrontal))
        # # # use erosion and dilation to remove some noise
        # # kernel = np.ones((7,7), np.uint8)
        # # anFrontal = cv2.erode(anFrontal.numpy(), kernel, iterations=1)
        # # anFrontal = cv2.dilate(anFrontal, kernel, iterations=1)
        # anFrontal = np.array(anFrontal).astype(np.uint8)

        # #### Lateral ####
        # # initialize SIFT
        # sift = cv2.SIFT_create()
        # # find keypoints and descriptors in each image
        # keypoints_plain, descriptors_plain = sift.detectAndCompute(plainLateral,None)
        # keypoints_im, descriptors_im = sift.detectAndCompute(grayLateral,None)
        # # initialize brute force matcher
        # bf = cv2.BFMatcher(crossCheck=True)
        # # Match descriptors.
        # matches = bf.match(descriptors_plain,descriptors_im)
        # # find source and distance points
        # src_pts = np.float32([ keypoints_plain[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        # dst_pts = np.float32([ keypoints_im[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        # # Find Homography matrix
        # M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # # find shape of plain (without screws) image
        # rows, cols = plainLateral.shape
        # anLateral = cv2.warpPerspective(plainLateral,M,(rows,cols))#[:image.shape[0],:image.shape[1]]
        # if anLateral.shape[0] > grayLateral.shape[0]:
        #     anLateral= anLateral[:grayLateral.shape[0],:]
        # else:
        #     grayLateral = grayLateral[:anLateral.shape[0],:]
        # if anLateral.shape[1] > grayLateral.shape[1]:
        #     anLateral = anLateral[:,:grayLateral.shape[1]]
        # else:
        #     grayLateral = grayLateral[:,:anLateral.shape[1]]
        # assert anLateral.shape == grayLateral.shape, "Annotation and image should be same size"
        # # subtract image to extract screws
        # anLateral = (anLateral-grayLateral)
        # # threshold image
        # anLateral = np.where(anLateral>29, 1, 0).astype(np.uint8)
        # # # crop out background and pad to fit original shape
        # crop_size_x, crop_size_y = int(np.floor((anLateral.shape[0]*(4/5)))), int(np.floor((anLateral.shape[1]*(4/5))))
        # pad_x, pad_y = int((anLateral.shape[0]-(crop_size_x-1))/2), int((anLateral.shape[1]-(crop_size_y-1))/2)
        # anLateral = transforms.CenterCrop((crop_size_x,crop_size_y))(torch.tensor(anLateral))
        # anLateral = (transforms.Pad((pad_x,pad_y))(anLateral))
        # # # use erosion and dilation to remove some noise
        # # kernel = np.ones((7,7), np.uint8)
        # # anLateral = cv2.erode(anLateral.numpy(), kernel, iterations=1)
        # # anLateral = cv2.dilate(anLateral, kernel, iterations=1)
        # anLateral = np.array(anLateral).astype(np.uint8)

        ### FOR ANNOTATION OF BONE
        # # load plain image (without screws)
        # plain = cv2.imread('/work3/dgro/frontal.jpg')
        # plain = cv2.cvtColor(plain, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # # load image with bone mask
        # boneMask = cv2.imread('/work3/dgro/frontalBoneMask.png')
        # boneMask = cv2.cvtColor(boneMask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # boneMask = np.where(boneMask>0, 1, 0).astype(np.uint8)
        # # load image with screws
        # image = cv2.imread(image_path[0])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        # # initialize SIFT
        # sift = cv2.SIFT_create()
        # # find keypoints and descriptors in each image
        # keypoints_plain, descriptors_plain = sift.detectAndCompute(plain,None)
        # keypoints_im, descriptors_im = sift.detectAndCompute(image,None)
        # # initialize brute force matcher
        # bf = cv2.BFMatcher(crossCheck=True)
        # # Match descriptors.
        # matches = bf.match(descriptors_plain,descriptors_im)
        # # find source and distance points
        # src_pts = np.float32([ keypoints_plain[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        # dst_pts = np.float32([ keypoints_im[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        # # Find Homography matrix
        # M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # # find shape of plain (without screws) image
        # rows, cols = plain.shape
        # anScrew = cv2.warpPerspective(plain,M,(rows,cols))#[:image.shape[0],:image.shape[1]]
        # anBone = cv2.warpPerspective(boneMask,M,(rows,cols))
        # if anScrew.shape[0] > image.shape[0]:
        #     anScrew = anScrew[:image.shape[0],:]
        #     anBone = anBone[:image.shape[0],:]
        # else:
        #     image = image[:anScrew.shape[0],:]
        # if anScrew.shape[1] > image.shape[1]:
        #     anScrew = anScrew[:,:image.shape[1]]
        #     anBone = anBone[:,:image.shape[1]]
        # else:
        #     image = image[:,:anScrew.shape[1]]
        # assert anScrew.shape == image.shape, "Annotation and image should be same size"
        # # subtract image to extract screws
        # anScrew = (anScrew-image)
        # # threshold image
        # anScrew = np.where(anScrew>29, 1, 0).astype(np.uint8)
        # #_,an = cv2.threshold(an,29,an.max(),cv2.THRESH_BINARY)
        # # crop out background and pad to fit original shape
        # crop_size_x, crop_size_y = int(np.floor((anScrew.shape[0]*(3/4)))), int(np.floor((anScrew.shape[1]*(3/4))))
        # pad_x, pad_y = int((anScrew.shape[0]-(crop_size_x-1))/2), int((anScrew.shape[1]-(crop_size_y-1))/2)
        # anScrew = transforms.CenterCrop((crop_size_x,crop_size_y))(torch.tensor(anScrew))
        # anScrew = (transforms.Pad((pad_x,pad_y))(anScrew))
        # # use erosion and dilation to remove some noise
        # kernel = np.ones((7,7), np.uint8)
        # anScrew = cv2.erode(anScrew.numpy(), kernel, iterations=1)
        # anScrew = cv2.dilate(anScrew, kernel, iterations=1)
        # anScrew = np.where(anScrew>0, 1, 0).astype(np.uint8)
        # anBone = np.where(anBone>0, 1, 0).astype(np.uint8)
        # # if transformation is provided, transform both image and annotations
        # if self.transform:
        #     image = self.transform(image)
        #     anScrew = self.transform(anScrew)
        #     anBone = self.transform(anBone)
        # # stack images together - first channel is the bone, second channel is the screws
        # # shape = (n_channel, height, width)
        # an = np.stack([anBone,anScrew],axis = 0)
        # return image and annotation
        # score = self.scores[idx]
        images = imread_collection([image_path[0], image_path[1]], conserve_memory=True) # stack the frontal and lateral images
        
        # resize to 900x900 and crop
        transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((900,900)),
                                    ])

        images_resized = [transform(img) for img in images]
        images_crop = []
        images_crop += [np.array(images_resized[0])[50:650,200:800]/255.0]
        images_crop += [np.array(images_resized[1])[100:800,100:800]/255.0]

        if self.transform:
            images = torch.stack([self.transform(img) for img in images_crop], dim = 1) # transform both frontal and lateral images
        #     grayFrontal = self.transform(grayFrontal)
        #     grayLateral = self.transform(grayLateral)
        #     anFrontal = self.transform(anFrontal)
        #     anLateral = self.transform(anLateral)
        # grays = np.stack([grayFrontal,grayLateral],axis = 0)
        # ans = np.stack([anFrontal,anLateral],axis = 0)
        score = self.scores[idx]
        assess = self.assess[idx]
        return images.float(), score.float(), assess.float()#, grays, ans

    def __getscale__(self):
        '''Return scales (mean and std) for trainset'''
        return self.mean_, self.std_

def get_loader(repair_type, split, data_path = "/work3/dgro/Data/", batch_size=16, transform = None, num_workers=0):
    """Build and return a data loader."""
    
    dataset = SimulationData(repair_type, split, data_path, transform, train_size = 0.8, test_size = 0.2, seed = 8)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    print('Finished loading dataset.')
    return data_loader
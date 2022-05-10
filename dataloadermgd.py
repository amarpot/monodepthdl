from pathlib import Path
#from numpy.linalg import inv

import numpy
import random
import torch
import numpy as np
import h5py
from PIL import Image

from numpy import zeros, newaxis


from torch.utils.data import Dataset, DataLoader, random_split

MEGADEPTH_ROOT = Path('/vol/vssp/datasets/mixedmode/Megadepth')
SCENE_ROOT = Path('/vol/research/monodepth_landmarks/python_fw/defeat/datasets')

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    print("target")
    
    #target = torch.LongTensor(target)
    #print(target)
    
    return [data, target]

def load_depth(file):
    """Load dense depth map.

    :param file: (PathLike) File to load (without MegaDepth root).
    :return: (ndarray) (h, w) Dense depth map.
    """
    with h5py.File(MEGADEPTH_ROOT/file, 'r') as f:
        depth = np.array(f['/depth']) 
        #depth = depth.resize(400,400) #use subplot
        #depth = depth[:, :, newaxis]
        #depth = np.tile(depth, (1, 1, 3))
        #print("shaped :", depth.shape)
        return depth




def process_image(image):
    
    #image = image.resize((400,400))
    #image = image.resize((400,400))
    image = np.array(image, dtype=np.float) / 255
    print("shape1", image.shape)
    return image


def get_scenes(mode):
    """Get file containing train/test scenes."""
    return SCENE_ROOT / 'splits' / f'{mode}_scenes.txt'

def load_scene_info(scene):
    
    return np.load(MEGADEPTH_ROOT/'scene_info'/f'{scene}.0.npz', allow_pickle=True)

def load_image(file):
    """Load PIL image by prepending the MegaDepth root."""
    #print(file)
    return Image.open(MEGADEPTH_ROOT/file)


def get_valid_pairs(scene_info, min_overlap, max_overlap, max_scale):
    """Get indexes of pairs satisfying overlap and scale change requirements.

    :param scene_info: (npz|dict) Scene information (loaded as above).
    :param min_overlap: (float) Minimum percentage overlap between images.
    :param max_overlap: (float) Maximum percentage overlap between images.
    :param max_scale: (float) Maximum change in scale between images.
    :return: (ndarray) (n, 2) Valid pairs as (idx1, idx2).
    """
    valid = (scene_info['overlap_matrix'] >= min_overlap) & \
            (scene_info['overlap_matrix'] <= max_overlap) & \
            (scene_info['scale_ratio_matrix'] <= max_scale)
  
    pairs = np.stack(np.where(valid), axis=-1)
    #print(pairs)
   

    return pairs


class MegaDepth(Dataset):
    def __init__(self, scene,transform = None):
        self.scene = scene
        self.scene_info = load_scene_info(self.scene) #pass scene to load_scene_info
        self.image_paths = self.scene_info['image_paths'] #obtain image paths
        self.check = get_valid_pairs(self.scene_info, 0.5, 0.9, np.inf)
        self.depth_paths = self.scene_info['depth_paths']
        self.camera_intrinsics = self.scene_info['intrinsics']
        self.poses_paths = self.scene_info['poses']
       


    def __getitem__(self, index):
       
        while (self.image_paths[index] == None):
            index = index + 1
        
        
        image = process_image(load_image(self.image_paths[index])) #load image
        print("shape", image.shape)
        depths = load_depth(self.depth_paths[index]) #load depth
        
        print("depth_shape", depths.shape)
        poses = self.poses_paths[index]
        print("poses", poses)
        self.k = self.camera_intrinsics[index]
        self.k = np.pad(self.k, [(0, 1), (0, 1)], mode='constant')
        self.k[3][3] = 1
        self.inv_k = inverse = numpy.linalg.inv(self.k)
        #self.inv_k = self.k
        print("self.k",self.k)
        print("self.inv_k",self.inv_k)
        self.support_frames = (-1,1)
        
        spf = []     #array to store support frames
        #print(self.check)
      
        for x in self.check: #iterate through the pairs returned by get_valid_pairs
            #print(x[0])
            if x[0] == index and self.image_paths[x[1]] != None:
                spf.append(x[1]) #append images that satisfy the above condition (check if the pairs correspond to index and the other value is a valid image
                
            if x[1] == index and self.image_paths[x[0]] != None:
                spf.append(x[0])
                
          
        while not spf:     #if no support frames move on to next index and find pairs 
            index = index + 1 
            for x in self.check:
                
                if x[0] == index and self.image_paths[x[1]] != None:
                    spf.append(x[1])
                    
                if x[1] == index and self.image_paths[x[0]] != None:
                    spf.append(x[0])
                    
    
 
        rand1 = random.randint(0, len(spf) - 1) #find random value to choose support frame
        rand2 = random.randint(0, len(spf) - 1)
        
        while (rand1 == rand2):
            rand1 = random.randint(0, len(spf) - 1) #find random value to choose support frame
            rand2 = random.randint(0, len(spf) - 1)   
            if (len(spf) - 1) < 3:
                break 

        sup1 = spf[rand1] #assign support frame
        sup2 = spf[rand2]
     
        support_iter = [sup1, sup2]
        self.support_images = []
        for z in support_iter: #load support frames
            a = process_image(load_image(self.image_paths[z]))
            self.support_images.append(a)
        show(image,depths,self.support_images)
        #a = np.concatenate((image,image), axis = 1)
        #b = np.concatenate(self.support_images, axis = 1)
        #c = np.concatenate((a,b) , axis = 0) #concatenate images
        print("frames", np.array(self.support_frames))
        return image,self.support_images, depths, (self.k, self.inv_k), np.array(self.support_frames)
    
    def __len__(self):
        li = len(self.image_paths)
        return li

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def show(image,depths,support_images):
    
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(image), ax2.imshow(depths), ax3.imshow(support_images[0]), ax4.imshow(support_images[1])
    #a = np.concatenate((image,image), axis = 1)
    #b = np.concatenate(support_images, axis = 1)
    #c = np.concatenate((a,b) , axis = 0) #concatenate images
    #plt.imshow(a)
    plt.pause(1)
plt.show()








file_path = get_scenes('test')
f = open(file_path,'r')
content = f.read()
content_list = content.splitlines()
f.close


ds = []

for scene in content_list: 
    dataset = MegaDepth(scene)
    print("dataset: ", dataset)
    ds.append(dataset)
    

#file_path = get_scenes('train')
#f = open(file_path,'r')
#content = f.read()
#content_list = content.splitlines()
#f.close

#ds2 = []         

#for scene in content_list: 
    #dataset2 = MegaDepth(scene)
    #print("dataset: ", dataset2)
    #ds2.append(dataset2)
    #break

tot = torch.utils.data.ConcatDataset(ds)

#tot2 = torch.utils.data.ConcatDataset(ds2)

#print("len", len(tot))

dataloader = DataLoader(dataset=tot, batch_size=8, collate_fn=my_collate,  shuffle =True, pin_memory=True)
        


dataiter = next(iter(dataloader))  

          



#dataloader2 = DataLoader(dataset=tot2, batch_size=1, shuffle =True)
        
#dataiter2 = next(iter(dataloader2))  





#iterate through all the scenes.DONEprint
#in built python/numpy generate a random number for support frame generate randomly and then add to the image output (refer to monodepth.py) DONE
#load 145 50 scenes for training/testing respectitively. DONE
#use the overlap matrix to compare each value to the overlap
#instrinsics and extrinsics area saved as 4 by 4 (converted 3 by 3 to 4 by 4) DONE
#Load poses DONE
#check shape of all images
#return them as should be

#0133 should be excluded PROBLEM SOLVED
#0122 or 0121
#0430


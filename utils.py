import scipy.io as scio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
import numpy as np
from PIL import Image
def load_data(path):
    dataFile = path + "/YoutubePose/YouTube_Pose_dataset.mat"
    data = scio.loadmat(dataFile)
    data = data['data'][0] # now data will be a 50 len array
    data_rearrange = []
    for d in data:
        dict = {}
        dict['url'] = d[0][0]
        dict['videoname'] = d[1][0]
        dict['locs'] = d[2]
        dict['frameids'] = d[3][0]
        dict['label_names'] = d[4]
        dict['crop'] = d[6][0]
        dict['scale'] = d[5][0][0]
        dict['isYouTubeSubset'] = d[8][0]
        dict['origRes'] = d[7][0]
        data_rearrange.append(dict)
        
    paths, labels = [],[]
    imagepath = path + "/YoutubePose/GT_frames/"
    for dic in data_rearrange:
        frameids = dic['frameids']
        locs = dic['locs']
        directory = imagepath + dic['videoname']
        t_path = [directory + "/frame_" + "%06d" % i + ".jpg" for i in frameids]
        paths.extend(t_path)
        t_locs = np.zeros([100,7,2])
        for i in range(100):
            for j in range(7):
                for k in range(2):
                    t_locs[i][j][k] = locs[k][j][i]
        labels.extend(t_locs)
    
    return data_rearrange,paths,labels

new_w=320
new_h=180

preprocess = transforms.Compose([
    #transform all image into 16:9
    transforms.Resize(size=(new_h,new_w)),
    transforms.ToTensor()
    
])

def default_loader(path):
    img_pil = Image.open(path)
    ow,oh=img_pil.size
    img_tensor = preprocess(img_pil)
    return img_tensor,ow,oh

class trainset(Dataset):
    def __init__(self, images, labels, loader = default_loader):
        self.images = images
        self.target = labels
        self.loader = loader
        
    def __getitem__(self,index):
        fn = self.images[index]
        img,ow,oh = self.loader(fn)
        #print(ow,oh)
        #print(img.shape)
        target = self.target[index]
        #print(target)
        target = target/np.array([ow/new_w,oh/new_h])
        #print(target)
        return img,target

    def __len__(self):
        return len(self.images)

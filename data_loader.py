import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, rotate

class ImageDatasetFromPt(Dataset):
    def __init__(self, data_path, preproc, angle = 0, blur = 0, noise_level=0, cnt = 20, cuda=True):
        super(ImageDatasetFromPt, self).__init__()
        self.device         = 'cuda' if cuda else 'cpu'
        self.data_dir       = data_path
        data                = torch.load(data_path, map_location=lambda storage, loc: storage)
        self.img            = []
        self.label          = []
        self.angle          = angle
        self.blur           = blur
        self.noise_level    = noise_level

        for i in range(1, 27):
            start_index     = min(np.where(data[1].data.numpy() == i)[0])
            self.img.extend(data[0].data.numpy()[start_index:start_index + cnt].tolist())
            self.label.extend(data[1].data.numpy()[start_index:start_index + cnt].tolist())

        self.preproc  = preproc

    def get_transformer(self):
        return self.preproc

    def __getitem__(self, index):
        image               = np.array(self.img[index])
        label               = self.label[index]

        if self.angle != 0:
            image           = rotate(image, self.angle)
        if self.blur >0 :
            image           = gaussian_filter(image, self.blur)
        if self.noise_level!=0:
            image           = np.array(image, dtype='float64')
            image          += self.noise_level*np.random.normal(0, (image.max() - image.min())/6., image.shape)

        image[image<0]      = 0
        image[image>255]    = 255
        image               = image/image.max()
        image               = Image.fromarray(np.array(image, dtype='float64'))
        image               = self.preproc(image)
        label               = torch.tensor(label).to(self.device)
        image               = torch.tensor(image).to(self.device)
        sample              = {'input': image, 'target': label, 'fname': str(index), 'params': ''}
        return sample

    def __len__(self):
        return len(self.label)

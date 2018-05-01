# Class for reading training and validation data
import numpy as np
import os
import skimage
import skimage.transform
import skimage.io
import random

def read_single_mask(filepath):
    '''
    Read single mask from directory and tranform to categorical
    '''
    mask = skimage.io.imread(filepath)
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    mask_trans = np.empty((512, 512), dtype=int)
    mask_trans[mask == 3] = 0  # (Cyan: 011) Urban land 
    mask_trans[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    mask_trans[mask == 5] = 2  # (Purple: 101) Rangeland 
    mask_trans[mask == 2] = 3  # (Green: 010) Forest land 
    mask_trans[mask == 1] = 4  # (Blue: 001) Water 
    mask_trans[mask == 7] = 5  # (White: 111) Barren land 
    mask_trans[mask == 0] = 6  # (Black: 000) Unknown 
    return mask_trans

def mask_to_rgb(labels_pred, filepath):
    '''
    Convert masks into RGB images and save to the filepath
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown

class Data_Reader:
    def __init__(self, ImageDir, GTLabelDir='', BatchSize=1, Suffle=True, img_size=256):
        self.NumFiles = 0
        self.Epoch = 0
        self.itr = 0
        # Image directory
        self.Image_Dir = ImageDir
        if GTLabelDir == '':
            self.ReadLabels=False
        else:
            self.ReadLabels=True
        self.Label_Dir = GTLabelDir
        self.OrderedFiles=[]
        # Read list of all files
        self.OrderedFiles += [each for each in os.listdir(self.Image_Dir) if each.endswith('.jpg')] # Get list of training images
        self.BatchSize = BatchSize
        self.NumFiles = len(self.OrderedFiles)
        self.OrderedFiles.sort()
        self.SuffleBatch()
        self.img_size = img_size
    
    def SuffleBatch(self):
        Sf = np.array(range(np.int32(np.ceil(self.NumFiles / self.BatchSize) + 1))) * self.BatchSize
        random.shuffle(Sf)
        self.SFiles = []
        for i in range(len(Sf)):
            for k in range(self.BatchSize):
                if Sf[i] + k < self.NumFiles:
                    self.SFiles.append(self.OrderedFiles[Sf[i] + k])
    
    def ReadAndAugmentNextBatch(self):
        ### End of an epoch
        if self.itr >= self.NumFiles:
            self.itr = 0
            self.SuffleBatch()
            self.Epoch += 1
        batch_size = np.min([self.BatchSize, self.NumFiles - self.itr])
        Sy = Sx = 0
        XF = YF = 1
        Cry = 1
        Crx = 1
        ### Resize Factor
        if np.random.rand() < 1:
            YF = XF = 0.3 + np.random.rand() * 0.7
        ### Stretch image
        if np.random.rand() < 0.8:
            if np.random.rand() < 0.5:
                XF *= 0.5 + np.random.rand() * 0.5
            else:
                YF *= 0.5 + np.random.rand() * 0.5
        ### Crop Image
        if np.random.rand() < 0.0:
            Cry = 0.7 + np.random.rand() * 0.3
            Crx = 0.7 + np.random.rand() * 0.3
        ### Augument Images and labeles
        for f in range(batch_size):
            #### Read image and labels from files
            Img = skimage.io.imread(self.Image_Dir + '/' + self.SFiles[self.itr])
            Img = Img[:,:,0:3]
            #### In HW3, image file name is 'xxxx_sat.jpg', while mask file name is 'xxxx_mask.png'
            LabelName = self.SFiles[self.itr][0:-7] + 'mask.png'
            if self.ReadLabels:
                Label = read_single_mask(self.Label_Dir + '/' + LabelName)
            self.itr += 1
            #### Set batch size at the first time
            if f == 0:
                Sy, Sx, d = Img.shape
                Sy *= YF
                Sx *= XF
                Cry *= Sy
                Crx *= Sx
                Sy = np.int32(Sy)
                Sx = np.int32(Sx)
                Cry = np.int32(Cry)
                Crx = np.int32(Crx)
                Images = np.zeros([batch_size, Cry, Crx, 3], dtype=np.float)
                if self.ReadLabels:
                    Labels = np.zeros([batch_size, Cry, Crx, 1], dtype=np.int)
                ##### For HW3, produce list of mask file names ('xxxx_mask.png') here
                LabelNames = []

            #### Resize and strecth image and labels
            Img = skimage.transform.resize(Img,
                                           [self.img_size, self.img_size],
                                           mode='reflect',
                                           preserve_range=True)
            if self.ReadLabels:
                Label = skimage.transform.resize(Label,
                                                 [self.img_size, self.img_size],
                                                 mode='reflect',
                                                 preserve_range=True).astype(int)

            #### Crop Image
            MinOccupancy = 501
            if not (Cry == Sy and Crx == Sx):
                for u in range(501):
                    MinOccupancy -= 1
                    Xi=np.int32(np.floor(np.random.rand() * (Sx - Crx)))
                    Yi=np.int32(np.floor(np.random.rand() * (Sy - Cry)))
                    if np.sum(Label[Yi:(Yi+Cry), Xi:(Xi+Crx)] > 0) > MinOccupancy:
                        Img=Img[Yi:(Yi+Cry), Xi:(Xi+Crx),:]
                        if self.ReadLabels:
                            Label = Label[Yi:(Yi+Cry), Xi:(Xi+Crx)]
                        break
            #### Mirror Image
            if random.random() < 0.5:
                Img=np.fliplr(Img)
                if self.ReadLabels:
                    Label = np.fliplr(Label)
            #### Agument color of Image
            Img = np.float32(Img)
            if np.random.rand() < 0.8:
                ##### Play with shade
                Img *= 0.4 + np.random.rand() * 0.6
            if np.random.rand() < 0.4:
                ##### Turn to grey
                Img[:, :, 2] = Img[:, :, 1] = Img[:, :, 0] = Img[:,:,0] = Img.mean(axis=2)
            if np.random.rand() < 0.0:
                ##### Play with color
                if np.random.rand() < 0.6:
                    for i in range(3):
                        Img[:, :, i] *= 0.1 + np.random.rand()
                ##### Add Noise
                if np.random.rand() < 0.2:
                    Img *=np.ones(Img.shape) * 0.95 + np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2]) * 0.1
            Img[Img > 255] = 255
            Img[Img < 0] = 0
            #### Add images and labels to to the batch
            Images[f] = Img
            if self.ReadLabels:
                Labels[f,:,:,0] = Label
            LabelNames.append(LabelName)
        ### Return aumented images and labels
        if self.ReadLabels:
            return Images, Labels, LabelNames
        else:
            return Images, LabelNames
    
    def ReadNextBatchClean(self):
        ### End of an epoch
        if self.itr >= self.NumFiles: 
            self.itr = 0
            #self.SuffleBatch()
            self.Epoch += 1
        batch_size = np.min([self.BatchSize, self.NumFiles - self.itr])
        for f in range(batch_size):
            # print('    Reading' + self.Image_Dir + '/' + self.OrderedFiles[self.itr])
            #### Read image and labels from files
            Img = skimage.io.imread(self.Image_Dir + '/' + self.OrderedFiles[self.itr])
            # print('    Before: Img.shape = %s' % (Img.shape,))
            Img = Img[:,:,0:3]
            # print('    After: Img.shape = %s' % (Img.shape,))
            #### In HW3, image file name is 'xxxx_sat.jpg', while mask file name is 'xxxx_mask.png'
            LabelName = self.OrderedFiles[self.itr][0:-7] + 'mask.png'
            if self.ReadLabels:
                Label = read_single_mask(self.Label_Dir + '/' + LabelName)
                # print('    Label:')
                # print(Label)
            self.itr += 1
            #### Set batch size at the first time
            if f == 0:
                Sy, Sx, Depth = Img.shape
                Images = np.zeros([batch_size, self.img_size, self.img_size, 3], dtype=np.float)
                if self.ReadLabels:
                    Labels = np.zeros([batch_size, self.img_size, self.img_size, 1], dtype=np.int)
                ##### For HW3, produce list of mask file names ('xxxx_mask.png') here
                LabelNames = []
            #### Resize image and labels
            Img = skimage.transform.resize(Img,
                                           [self.img_size, self.img_size],
                                           mode='reflect',
                                           preserve_range=True)
            if self.ReadLabels:
                Label = skimage.transform.resize(Label,
                                                 [self.img_size, self.img_size],
                                                 mode='reflect',
                                                 preserve_range=True).astype(int)
            #### Load image and label to batch
            Images[f] = Img
            if self.ReadLabels:
                Labels[f, :, :, 0] = Label
            LabelNames.append(LabelName)
        ### Return images and labels
        if self.ReadLabels:
            return Images, Labels, LabelNames
        else:
            return Images, LabelNames

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class makeDataset(Dataset):
    def __init__(self, dataset, labels, spatial_transform, seqLen=20):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.seqLen = seqLen

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        vid=cv2.VideoCapture(vid_name)
        ret, prev_frame = vid.read()
        p_frame_thresh = 20
        count=1
        index=0
        while(index<=self.seqLen):
            # Extract images
            ret, curr_frame = vid.read()
            diff = cv2.absdiff(curr_frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count > p_frame_thresh:
                count+=1
                img=Image.fromarray(curr_frame)
                inpSeq.append(self.spatial_transform(img.convert('RGB')))
            prev_frame = curr_frame
        print (count)
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
import numpy as np
from torch.utils import data
from pathlib import Path
from PIL import Image
import os

class BaseDataset(data.Dataset):
    def __init__(self, is_global, list_path, set_, data_name, image_size, labels_size, mean=(128, 128, 128)):
        self.set = set_
        self.is_global = is_global
        self.data_name = data_name
        self.list_path = list_path
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean_rgb = mean
        self.mean_gray = mean[0]
        list_file_name = self.set + '.txt'        
        list_file_path = os.path.join(self.list_path, list_file_name)

        with open(list_file_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        self.files = []
        for name in self.img_ids:
            reference_file = os.path.join(name, 'reference.png')
            overlaid_file = os.path.join(name, 'overlaid.png')
            gt_file = os.path.join(name, 'gt.png')
            
            left_gray_patch_file = os.path.join(name, 'left_patch_gray_angle.png')
            right_gray_patch_file = os.path.join(name, 'right_patch_gray_angle.png')
            mouth_gray_patch_file = os.path.join(name, 'mouth_patch_gray_angle.png')

            left_rgb_gt_file = os.path.join(name, 'left_patch_rgb_gt.png')
            right_rgb_gt_file = os.path.join(name, 'right_patch_rgb_gt.png')
            mouth_rgb_gt_file = os.path.join(name, 'mouth_patch_rgb_gt.png')

            mask_file = os.path.join(name, 'mask.npy')

            if self.is_global:
                self.files.append((reference_file, overlaid_file, gt_file, 
                                left_gray_patch_file, left_rgb_gt_file, right_gray_patch_file, 
                                right_rgb_gt_file, mouth_gray_patch_file, mouth_rgb_gt_file,
                                mask_file, name))
            else:
                # self.files.append((reference_file, overlaid_file, label_file, 
                #                     left_gray_patch_file, right_gray_patch_file, mouth_gray_patch_file,
                #                     left_rgb_gt_file, right_rgb_gt_file, mouth_rgb_gt_file, mask_file, name))
                self.files.append((reference_file, gt_file, 
                                    left_gray_patch_file, left_rgb_gt_file, right_gray_patch_file, right_rgb_gt_file,
                                    mouth_gray_patch_file, mouth_rgb_gt_file,
                                    mask_file, name))

        # if max_iters is not None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids))) # ???
        # self.files = []

        # for name in self.img_ids:
        #     if self.data_name != 'MESH':
        #         img_file, label_file = self.get_metadata(name)
        #         self.files.append((img_file, label_file, name))
        #     else:
        #         img_file = self.get_metadata(name)
        #         self.files.append((img_file, name))


    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess_gray(self, image):
        image -= self.mean_gray
        image = image/128.
        image = image[..., np.newaxis] # Expand the dim of the gray image
        final_image = image.transpose((2, 0, 1)) # to [c h w]
        return final_image

    def preprocess_rgb(self, image):
        '''
        Standardazation: (x-mean)/128.
        '''
        # image = image[:, :, ::-1] # change to BGR???

        # Standardazation
        image -= self.mean_rgb

        # # Normalization: [0, 1]
        # image_max = np.max(image)
        # image_min = np.min(image)
        # image_range = image_max - image_min
        # image = (image - image_min) / image_range
        
        # Normalization: [-1, 1]
        image = image/128.
        final_image = image.transpose((2, 0, 1)) # to [c h w]
        return final_image

    def get_image(self, file):
        return _load_img(file, (self.image_size, self.image_size), Image.BICUBIC, rgb=True) 

    def get_labels(self, file):
        return _load_img(file, (self.image_size, self.image_size), Image.NEAREST, rgb=False) 

    def get_patch(self, file):
        img = Image.open(file) # Gray images with channel of 3
        img = img.convert('L') # Convert the channel number to 1
        # img = img.resize(new_size) # Resize the patch according to the mask size
        img = img.resize((self.image_size, self.image_size)) # Resize the patch to the square size
        img_np = np.asarray(img).astype(np.float32)
        return img_np

    # def get_gt_gray_patch(self, file, mask_pos_np):
    #     img = Image.open(file)
    #     img = img.convert('L')
    #     v_min = np.min(mask_pos_np[0])
    #     v_max = np.max(mask_pos_np[0])+1
    #     u_min = np.min(mask_pos_np[1])
    #     u_max = np.max(mask_pos_np[1])+1

    #     new_img = img.crop((u_min, v_min, u_max, v_max))
    #     new_img = new_img.resize((self.image_size, self.image_size))
    #     new_img_np = np.asarray(new_img).astype(np.float32)
    #     new_img_np -= self.mean_gray
    #     new_img_np /= 128.
    #     new_img_np = new_img_np[..., np.newaxis]
    #     new_img_np = new_img_np.transpose((2, 0, 1))

    #     return new_img_np

    def get_mask(self, file):
        raw_mask_np = np.load(file)
        raw_mask_im = Image.fromarray(np.uint8(raw_mask_np)) # Convert the value of -1 to 255
        resized_mask_im = raw_mask_im.resize((self.image_size, self.image_size), Image.NEAREST) # Do not change the mask values
        
        return np.asarray(resized_mask_im)
    
def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    # return np.asarray(img, np.float32)
    return img
    
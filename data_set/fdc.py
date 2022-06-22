from data_set.base_dataset import BaseDataset
import numpy as np

class FDCDataset(BaseDataset):
    def __init__(self, list_path, set='train', data_name='FDC',
                image_size=256,
                mean=None):
        super().__init__(list_path, set, data_name, image_size, mean)

        self.data_name = data_name
        

    def __getitem__(self, index):
        img_file, label_file, left_gray_file, right_gray_file, mouth_gray_file, left_rgb_file, right_rgb_file, mouth_rgb_file, name = self.files[index]
        
        # Label is the gt rgb image
        label = self.get_image(label_file)
        label = self.preprocess(label)
        image = self.get_image(img_file)
        image = self.preprocess(image)

        left_gray = self.get_image(left_gray_file)
        left_gray = self.preprocess(left_gray)

        right_gray = self.get_image(right_gray_file)
        right_gray = self.preprocess(right_gray)

        mouth_gray = self.get_image(mouth_gray_file)
        mouth_gray = self.preprocess(mouth_gray)

        left_gt = self.get_image(left_rgb_file)
        left_gt = self.preprocess(left_gt)

        right_gt = self.get_image(right_rgb_file)
        right_gt = self.preprocess(right_gt)

        mouth_gt = self.get_image(mouth_rgb_file)
        mouth_gt = self.preprocess(mouth_gt)

        return image.copy(), label.copy(), left_gray.copy(), right_gray.copy(), mouth_gray.copy(), left_gt.copy(), right_gt.copy(), mouth_gt.copy(), np.array(image.shape), name
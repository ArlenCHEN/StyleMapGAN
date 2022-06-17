from data_set.base_dataset import BaseDataset
import numpy as np

class FDCDataset(BaseDataset):
    def __init__(self, list_path, set='train', data_name='FDC',
                image_size=256,
                mean=None):
        super().__init__(list_path, set, data_name, image_size, mean)

        self.data_name = data_name
        

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.preprocess(label)
        image = self.get_image(img_file)
        image = self.preprocess(image)

        return image.copy(), label.copy(), np.array(image.shape), name
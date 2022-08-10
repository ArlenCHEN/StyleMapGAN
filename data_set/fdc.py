from copy import deepcopy
from data_set.base_dataset import BaseDataset
import numpy as np
from PIL import Image

class FDCDataset(BaseDataset):
    def __init__(self, is_global, is_equal, list_path, local_transforms, gray_ref_transforms, set='train', data_name='FDC',
                image_size=256,
                mean=None):
        super().__init__(is_global, is_equal, list_path, set, data_name, image_size, mean)

        self.data_name = data_name
        self.local_transforms = local_transforms
        self.gray_ref_transforms = gray_ref_transforms
        

    def __getitem__(self, index):
        if self.is_global:
            if self.is_equal:
                ref_file, gt_file, left_gray_file, left_rgb_gt_file, right_gray_file, right_rgb_gt_file, mouth_gray_file, mouth_rgb_gt_file, mask_file, name = self.files[index]

                ref_rgb = self.get_image(ref_file)
                ref_rgb_np = np.asarray(ref_rgb)
                ref_rgb_np = self.preprocess_rgb(ref_rgb_np)

                mask_np = self.get_mask(mask_file)
                
                gt_rgb = self.get_image(gt_file)
                gt_rgb_np = np.asarray(gt_rgb)
                overlaid_np = deepcopy(gt_rgb_np)

                mask_left_pos = np.where(mask_np==1)
                left_v_min = np.min(mask_left_pos[0])
                left_v_max = np.max(mask_left_pos[0])
                left_u_min = np.min(mask_left_pos[1])
                left_u_max = np.max(mask_left_pos[1])

                left_eye_h = left_v_max - left_v_min 
                left_eye_w = left_u_max - left_u_min

                left_gray_img = Image.open(left_gray_file) # The gray image channel is 3
                left_gray_img = left_gray_img.resize((left_eye_w, left_eye_h), Image.BILINEAR)
                left_gray_np = np.asarray(left_gray_img)
                overlaid_np[left_v_min:left_v_max, left_u_min:left_u_max] = left_gray_np

                mask_right_pos = np.where(mask_np==2)
                right_v_min = np.min(mask_right_pos[0])
                right_v_max = np.max(mask_right_pos[0])
                right_u_min = np.min(mask_right_pos[1])
                right_u_max = np.max(mask_right_pos[1])

                right_eye_h = right_v_max - right_v_min 
                right_eye_w = right_u_max - right_u_min

                right_gray_img = Image.open(right_gray_file) # The gray image channel is 3
                right_gray_img = right_gray_img.resize((right_eye_w, right_eye_h), Image.BILINEAR)
                right_gray_np = np.asarray(right_gray_img)
                overlaid_np[right_v_min:right_v_max, right_u_min:right_u_max] = right_gray_np

                mask_mouth_pos = np.where(mask_np==3)
                mouth_v_min = np.min(mask_mouth_pos[0])
                mouth_v_max = np.max(mask_mouth_pos[0])
                mouth_u_min = np.min(mask_mouth_pos[1])
                mouth_u_max = np.max(mask_mouth_pos[1])

                mouth_eye_h = mouth_v_max - mouth_v_min 
                mouth_eye_w = mouth_u_max - mouth_u_min

                mouth_gray_img = Image.open(mouth_gray_file) # The gray image channel is 3
                mouth_gray_img = mouth_gray_img.resize((mouth_eye_w, mouth_eye_h), Image.BILINEAR)
                mouth_gray_np = np.asarray(mouth_gray_img)
                overlaid_np[mouth_v_min:mouth_v_max, mouth_u_min:mouth_u_max] = mouth_gray_np
                overlaid_rgb_np = self.preprocess_rgb(overlaid_np)
                gt_rgb_np = self.preprocess_rgb(gt_rgb_np)

                return ref_rgb_np.copy(), overlaid_rgb_np.copy(), gt_rgb_np.copy(), mask_np.copy()
            else:
                ref_file, overlaid_file, gt_file, mask_file, name = self.files[index]
                
                ref_rgb = self.get_image(ref_file)
                ref_rgb_np = np.asarray(ref_rgb)
                ref_rgb_np = self.preprocess_rgb(ref_rgb_np)
                
                overlaid_rgb = self.get_image(overlaid_file)
                overlaid_rgb_np = np.asarray(overlaid_rgb)
                overlaid_rgb_np = self.preprocess_rgb(overlaid_rgb_np)

                gt_rgb = self.get_image(gt_file)
                gt_rgb_np = np.asarray(gt_rgb)
                gt_rgb_np = self.preprocess_rgb(gt_rgb_np)
                
                # Obtain the resized numpy mask file
                mask_np = self.get_mask(mask_file)
                
                return ref_rgb_np.copy(), overlaid_rgb_np.copy(), gt_rgb_np.copy(), mask_np.copy()
        else: # Local based method
            ref_file, gt_file, left_gray_file, left_rgb_gt_file, right_gray_file, right_rgb_gt_file, mouth_gray_file, mouth_rgb_gt_file, mask_file, name = self.files[index]
            mask_np = self.get_mask(mask_file)

            ref_rgb = self.get_image(ref_file)
            ref_gray = ref_rgb.convert('L')
            ref_rgb_np = np.asarray(ref_rgb)
            ref_rgb_np = self.preprocess_rgb(ref_rgb_np) # Return this

            if self.gray_ref_transforms is not None:
                ref_rgb_img_1 = Image.open(ref_file)
                ref_gray = self.gray_ref_transforms(ref_rgb_img_1)
            
            ref_gray_np = np.asarray(ref_gray)
            ref_gray_np = self.preprocess_gray(ref_gray_np) # Return this
            ref_mask = np.ones(mask_np.shape)
            ref_mask[mask_np!=255] = 0
            ref_gray_np = np.multiply(ref_mask, ref_gray_np).astype(np.float32) # Return this if the ref img needs to be masked
            
            gt_rgb = self.get_image(gt_file)
            gt_rgb_np = np.asarray(gt_rgb)
            gt_rgb_np = self.preprocess_rgb(gt_rgb_np) # Return this

            gt_gray = gt_rgb.convert('L')
            gt_gray_np = np.asarray(gt_gray)
            gt_gray_np = self.preprocess_gray(gt_gray_np) # Return this

            # ===================Left Patch======================
            # =========gt===========
            left_gray_gt_np = self.get_patch(left_rgb_gt_file)
            left_gray_gt_np = self.preprocess_gray(left_gray_gt_np)

            # =========raw===========
            if self.local_transforms is not None:
                left_img = Image.open(left_gray_file)
                left_gray = self.local_transforms(left_img)
                left_gray_np = np.asarray(left_gray)
            else:
                left_gray_np = self.get_patch(left_gray_file)
            left_gray_np = self.preprocess_gray(left_gray_np)

            # ===================Right Patch======================
            # =========gt===========
            right_gray_gt_np = self.get_patch(right_rgb_gt_file)
            right_gray_gt_np = self.preprocess_gray(right_gray_gt_np)

            # =========raw===========
            if self.local_transforms is not None:
                right_img = Image.open(right_gray_file)
                right_gray = self.local_transforms(right_img)
                right_gray_np = np.asarray(right_gray)
            else:
                right_gray_np = self.get_patch(right_gray_file)
            right_gray_np = self.preprocess_gray(right_gray_np)

            # ===================Mouth Patch======================
            # =========gt===========
            mouth_gray_gt_np = self.get_patch(mouth_rgb_gt_file)
            mouth_gray_gt_np = self.preprocess_gray(mouth_gray_gt_np)

            if self.local_transforms is not None:
                mouth_img = Image.open(mouth_gray_file)
                mouth_gray = self.local_transforms(mouth_img)
                mouth_gray_np = np.asarray(mouth_gray)
            else:
                mouth_gray_np = self.get_patch(mouth_gray_file)
            mouth_gray_np = self.preprocess_gray(mouth_gray_np)

            return ref_rgb_np.copy(), ref_gray_np.copy(), gt_rgb_np.copy(), gt_gray_np.copy(), left_gray_gt_np.copy(), left_gray_np.copy(), right_gray_gt_np.copy(), right_gray_np.copy(), mouth_gray_gt_np.copy(), mouth_gray_np.copy(), mask_np.copy()

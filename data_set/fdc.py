from data_set.base_dataset import BaseDataset
import numpy as np

class FDCDataset(BaseDataset):
    def __init__(self, is_global, list_path, set='train', data_name='FDC',
                image_size=256,
                mean=None):
        super().__init__(is_global, list_path, set, data_name, image_size, mean)

        self.data_name = data_name
        

    def __getitem__(self, index):
        if self.is_global:
            ref_file, overlaid_file, gt_file, left_gray_patch_file, right_gray_patch_file, mouth_gray_patch_file, mask_file, name = self.files[index]
            
            # Convert to rgb and resize the image
            # Return a pillow image
            ref_rgb = self.get_image(ref_file)
            # ref_gray = ref_rgb.convert('L')
            ref_rgb_np = np.asarray(ref_rgb, np.float32)
            ref_rgb_np = self.preprocess_rgb(ref_rgb_np)
            
            # ref_gray_np = np.asarray(ref_gray, np.float32)
            # ref_gray_np = ref_gray_np[..., np.newaxis]
            # ref_gray_np = self.preprocess_gray(ref_gray_np)
            
            overlaid_rgb = self.get_image(overlaid_file)
            overlaid_rgb_np = np.asarray(overlaid_rgb, np.float32)
            overlaid_rgb_np = self.preprocess_rgb(overlaid_rgb_np)

            overlaid_gray = overlaid_rgb.convert('L')
            overlaid_gray_np = np.asarray(overlaid_gray, np.float32)
            # overlaid_gray_np = np.repeat(overlaid_gray_np[:,:,np.newaxis], 3, axis=2)
            overlaid_gray_np = overlaid_gray_np[..., np.newaxis]
            overlaid_gray_np = self.preprocess_gray(overlaid_gray_np)
            
            gt_rgb = self.get_image(gt_file)
            gt_gray = gt_rgb.convert('L')
            gt_rgb_np = np.asarray(gt_rgb, np.float32)
            gt_rgb_np = self.preprocess_rgb(gt_rgb_np)

            gt_gray_np = np.asarray(gt_gray, np.float32)
            gt_gray_np = gt_gray_np[..., np.newaxis]
            # gt_gray_np = np.repeat(gt_gray_np[:,:,np.newaxis], 3, axis=2)
            gt_gray_np = self.preprocess_gray(gt_gray_np)
            
            # Obtain the resized numpy mask file
            mask_np = self.get_mask(mask_file)
            
            # Obtain the new size of the left patch
            mask_left_pos = np.where(mask_np==1) # Left part
            left_v_min = np.min(mask_left_pos[0])
            left_v_max = np.max(mask_left_pos[0])
            left_u_min = np.min(mask_left_pos[1])
            left_u_max = np.max(mask_left_pos[1])
            left_v_range = left_v_max - left_v_min
            left_u_range = left_u_max - left_u_min
            left_new_size = (left_v_range, left_u_range)
            left_gray_np = self.get_patch(left_gray_patch_file, left_new_size)
            left_gray_np = np.asarray(left_gray_np, np.float32)

            ## Input gray patches are treated as RGB input as the channel is 3

            # left_gray_np = left_gray_np[..., np.newaxis]
            # print('In fdc, left gray np shape: ', left_gray_np.shape)

            left_gray_np = self.preprocess_rgb(left_gray_np)

            mask_right_pos = np.where(mask_np==2) # Right part
            right_v_min = np.min(mask_right_pos[0])
            right_v_max = np.max(mask_right_pos[0])
            right_u_min = np.min(mask_right_pos[1])
            right_u_max = np.max(mask_right_pos[1])
            right_v_range = right_v_max - right_v_min
            right_u_range = right_u_max - right_u_min
            right_new_size = (right_v_range, right_u_range)
            right_gray_np = self.get_patch(right_gray_patch_file, right_new_size)
            right_gray_np = np.asarray(right_gray_np, np.float32)
            # right_gray_np = right_gray_np[..., np.newaxis]
            right_gray_np = self.preprocess_rgb(right_gray_np)

            mask_mouth_pos = np.where(mask_np==3) # Mouth part
            mouth_v_min = np.min(mask_mouth_pos[0])
            mouth_v_max = np.max(mask_mouth_pos[0])
            mouth_u_min = np.min(mask_mouth_pos[1])
            mouth_u_max = np.max(mask_mouth_pos[1])
            mouth_v_range = mouth_v_max - mouth_v_min
            mouth_u_range = mouth_u_max - mouth_u_min
            mouth_new_size = (mouth_v_range, mouth_u_range)
            mouth_gray_np = self.get_patch(mouth_gray_patch_file, mouth_new_size)
            mouth_gray_np = np.asarray(mouth_gray_np, np.float32)
            # mouth_gray_np = mouth_gray_np[..., np.newaxis]
            mouth_gray_np = self.preprocess_rgb(mouth_gray_np)

            # return ref_rgb_np.copy(), ref_gray_np.copy(), overlaid_gray_np.copy(), gt_rgb_np.copy(), gt_gray_np.copy(), mask_np.copy(), left_gray_np.copy(), right_gray_np.copy(), mouth_gray_np.copy(), np.array(ref_rgb_np.shape), name
            # return overlaid_rgb_np.copy(), overlaid_gray_np.copy(), gt_rgb_np.copy(), gt_gray_np.copy(), np.array(ref_rgb_np.shape), name
            return ref_rgb_np.copy(), overlaid_rgb_np.copy(), gt_rgb_np.copy(), mask_np.copy()
        else: # Local based method
            ref_file, gt_file, left_gray_file, left_rgb_gt_file, right_gray_file, right_rgb_gt_file, mouth_gray_file, mouth_rgb_gt_file, mask_file, name = self.files[index]

            ref_rgb = self.get_image(ref_file)
            ref_gray = ref_rgb.convert('L')
            ref_rgb_np = np.asarray(ref_rgb, np.float32)
            ref_rgb_np = self.preprocess_rgb(ref_rgb_np) # Return this
            
            ref_gray_np = np.asarray(ref_gray, np.float32)
            ref_gray_np = self.preprocess_gray(ref_gray_np) # Return this

            gt_rgb = self.get_image(gt_file)
            gt_gray = gt_rgb.convert('L')
            gt_rgb_np = np.asarray(gt_rgb, np.float32)
            gt_rgb_np = self.preprocess_rgb(gt_rgb_np) # Return this

            gt_gray_np = np.asarray(gt_gray, np.float32)
            gt_gray_np = self.preprocess_gray(gt_gray_np) # Return this
            
            mask_np = self.get_mask(mask_file)

            # ===================Left Patch======================
            # mask_left_pos = np.where(mask_np==1) 
            # left_gt_np = self.get_gt_gray_patch(gt_file, mask_left_pos)
            left_gray_gt_np = self.get_patch(left_rgb_gt_file)
            left_gray_gt_np = self.preprocess_gray(left_gray_gt_np)
            left_gray_np = self.get_patch(left_gray_file)
            left_gray_np = self.preprocess_gray(left_gray_np)
            # ===================Right Patch======================
            # mask_right_pos = np.where(mask_np==2)
            # right_gt_np = self.get_gt_gray_patch(gt_file, mask_right_pos)
            right_gray_gt_np = self.get_patch(right_rgb_gt_file)
            right_gray_gt_np = self.preprocess_gray(right_gray_gt_np)
            right_gray_np = self.get_patch(right_gray_file)
            right_gray_np = self.preprocess_gray(right_gray_np)
            # ===================Mouth Patch======================
            # mask_mouth_pos = np.where(mask_np==3)
            # mouth_gt_np = self.get_gt_gray_patch(gt_file, mask_mouth_pos)
            mouth_gray_gt_np = self.get_patch(mouth_rgb_gt_file)
            mouth_gray_gt_np = self.preprocess_gray(mouth_gray_gt_np)
            mouth_gray_np = self.get_patch(mouth_gray_file)
            mouth_gray_np = self.preprocess_gray(mouth_gray_np)

            return ref_rgb_np.copy(), ref_gray_np.copy(), gt_rgb_np.copy(), gt_gray_np.copy(), left_gray_gt_np.copy(), left_gray_np.copy(), right_gray_gt_np.copy(), right_gray_np.copy(), mouth_gray_gt_np.copy(), mouth_gray_np.copy(), mask_np.copy()





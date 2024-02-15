# https://github.com/IDEA-Research/DWPose
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .onnxdet import inference_detector
from .onnxpose import inference_pose
from segment_anything import SamPredictor, sam_model_registry

ModelDataPathPrefix = Path("./pretrained_weights")


class Wholebody:
    def __init__(self, device="cuda:0", use_sam=False):
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        onnx_det = ModelDataPathPrefix.joinpath("DWPose/yolox_l.onnx")
        onnx_pose = ModelDataPathPrefix.joinpath("DWPose/dw-ll_ucoco_384.onnx")

        self.session_det = ort.InferenceSession(
            path_or_bytes=onnx_det, providers=providers
        )
        self.session_pose = ort.InferenceSession(
            path_or_bytes=onnx_pose, providers=providers
        )

        if use_sam:
            sam_checkpoint = ModelDataPathPrefix.joinpath("SAM/sam_vit_h_4b8939.pth")
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
        else:
            self.sam_predictor = None

    def segment_mask_predict(self, oriImg_full, det_result, prompt=None):
        self.sam_predictor.set_image(oriImg_full)
        masks, scores, logits = self.sam_predictor.predict(box=det_result)
        # rank
        mask = masks[0,...]
        area = np.sum(mask)
        for i in range(masks.shape[0]):
            if np.sum(masks[i,...]) > area:
                area = np.sum(masks[i,...])
                mask = masks[i,...]

        return mask

    def __call__(self, oriImg, oriImg_full=None):
        det_result = inference_detector(self.session_det, oriImg)
        if oriImg_full is not None:
            assert self.sam_predictor is not None, "no sam detector found"
            ratio_xy = [oriImg_full.shape[1] / oriImg.shape[1], oriImg_full.shape[1] / oriImg.shape[1]]
            if len(det_result) == 0:
                det_result_full = [[0, 0, oriImg.shape[1], oriImg.shape[0]]]
            det_result_full = det_result.copy()
            for i in range(len(det_result)):
                det_result_full[i][0] = det_result[i][0] * ratio_xy[0]
                det_result_full[i][1] = det_result[i][1] * ratio_xy[1]
                det_result_full[i][2] = det_result[i][2] * ratio_xy[0]
                det_result_full[i][3] = det_result[i][3] * ratio_xy[1]

            mask = self.segment_mask_predict(oriImg_full, det_result_full[0, :])
            # update image
            oriImg_full[np.logical_not(mask), 0] = 255
            oriImg_full[np.logical_not(mask), 1] = 255
            oriImg_full[np.logical_not(mask), 2] = 255
            image_update = cv2.resize(oriImg_full, (oriImg.shape[1], oriImg.shape[0]))
            image_update = cv2.cvtColor(image_update, cv2.COLOR_RGB2BGR)
            oriImg = image_update
            #image_update = HWC3(image_update)
            #input_image = resize_image(input_image, detect_resolution)

        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        if oriImg_full is not None:
            return keypoints, scores, oriImg_full, mask

        return keypoints, scores

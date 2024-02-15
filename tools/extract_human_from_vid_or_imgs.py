import concurrent.futures
import os
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
from PIL import Image
import av

import numpy as np
from glob import glob


from src.dwpose import DWposeDetector
from src.utils.util import get_fps


def process_single_video(video_root, save_pure_root, save_pose_root, save_debug_root, video_path, detector):
    relative_path = os.path.relpath(video_path, video_root)
    out_pure_path = os.path.join(save_pure_root, relative_path)
    out_pose_path = os.path.join(save_pose_root, relative_path)
    out_debug_path = os.path.join(save_debug_root, relative_path)

    if os.path.exists(out_pure_path) or os.path.exists(out_pose_path):
        print('{} already exists'.format(out_pure_path))
        return
    out_pure_dir = Path(os.path.dirname(out_pure_path))
    out_pose_dir = Path(os.path.dirname(out_pose_path))
    out_debug_dir = Path(os.path.dirname(out_debug_path))
    if not out_pure_dir.exists():
        out_pure_dir.mkdir(parents=True, exist_ok=True)
    if not out_pose_dir.exists():
        out_pose_dir.mkdir(parents=True, exist_ok=True)
    if not out_debug_dir.exists():
        out_debug_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)
    container_in = av.open(video_path)
    stream_in = next(s for s in container_in.streams if s.type == "video")
    width, height = stream_in.width, stream_in.height

    # create output pure stream
    save_fmt = Path(out_pure_path).suffix
    assert save_fmt == ".mp4"
    #os.makedirs(os.path.dirname(out_path), exist_ok=True)
    codec = "libx264"

    container_pose = av.open(out_pose_path, "w")
    stream_pose = container_pose.add_stream(codec, rate=fps)
    stream_pose.width = width
    stream_pose.height = height

    if detector.use_sam:
        container_pure = av.open(out_pure_path, "w")
        stream_pure = container_pure.add_stream(codec, rate=fps)
        stream_pure.width = width
        stream_pure.height = height

        container_debug = av.open(out_debug_path, "w")
        stream_debug = container_debug.add_stream(codec, rate=fps)
        stream_debug.width = width
        stream_debug.height = height

    for packet in container_in.demux(stream_in):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            if detector.use_sam:
                result, score, image_pure, mask = detector(image)
                av_frame = av.VideoFrame.from_image(result)
                container_pose.mux(stream_pose.encode(av_frame))

                av_frame = av.VideoFrame.from_ndarray(image_pure)
                container_pure.mux(stream_pure.encode(av_frame))

                mask = mask.astype(np.uint8)*255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                av_frame = av.VideoFrame.from_ndarray(mask)
                container_debug.mux(stream_debug.encode(av_frame))
            else:
                result, score = detector(image)
                av_frame = av.VideoFrame.from_image(result)
                container_pose.mux(stream_pose.encode(av_frame))
            score = np.mean(score, axis=-1)

    container_pose.mux(stream_pose.encode())
    container_pose.close()

    if detector.use_sam:
        container_pure.mux(stream_pure.encode())
        container_pure.close()

        container_debug.mux(stream_debug.encode())
        container_debug.close()


def process_batch_videos(video_root, save_pure_root, save_pose_root, save_debug_root, video_lists, detector):
    for i, video_path in enumerate(video_lists):
        print(f'Processing video {i}/{len(video_lists)}')
        process_single_video(video_root, save_pure_root, save_pose_root, save_debug_root, video_path, detector)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', type=str)
    parser.add_argument('--save_pure_root', type=str)
    parser.add_argument('--save_pose_root', type=str)

    # Extract dwpose mp4 videos from raw videos
    # /path/to/video_dataset/${*/*/}${*.mp4} -> /path/to/video_dataset_dwpose/${*/*}/{*.mp4}
    # /path/to/video_dataset/${*/*/}${*.png} -> /path/to/video_dataset_dwpose/${*}/${*}.mp4
    parser.add_argument('--path_pattern', type=str, default='*')
    parser.add_argument('--name_pattern', type=str, default='*.mp4')

    parser.add_argument("-j", type=int, default=4, help="Num workers")

    args = parser.parse_args()
    num_workers = args.j

    conf = SimpleNamespace()

    if args.save_pure_root is None:
        args.save_pure_root = args.video_root + "_pure"

    if args.save_pose_root is None:
        args.save_pose_root = args.video_root + "_pose"

    save_debug_root = args.video_root + "_debug"
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    print(f"avaliable gpu ids: {gpu_ids}")

    assert 'mp4' in args.name_pattern
    video_lists = glob(os.path.join(args.video_root, args.path_pattern, args.name_pattern))
    if num_workers == 1:
        detector = DWposeDetector(use_sam=True)
        detector = detector.to("cuda:0")
        process_batch_videos(args.video_root, args.save_pure_root, args.save_pose_root, save_debug_root, video_lists, detector)
    else:
        batch_size = (len(video_lists) + num_workers - 1) // num_workers
        print(f"Num videos: {len(video_lists)} {batch_size = }")
        video_chunks = [
            video_lists[i: i + batch_size]
            for i in range(0, len(video_lists), batch_size)
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, chunk in enumerate(video_chunks):
                # init detector
                gpu_id = gpu_ids[i % len(gpu_ids)]
                detector = DWposeDetector()
                # torch.cuda.set_device(gpu_id)
                detector = detector.to(f"cuda:{gpu_id}")

                futures.append(
                    executor.submit(
                        args.video_root, args.save_pure_root, args.save_pose_root, save_debug_root, video_chunks, detector
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                future.result()





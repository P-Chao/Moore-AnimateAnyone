# -*- coding: utf-8 -*-
import concurrent.futures
import os
import random
from pathlib import Path
from PIL import Image
import av

import numpy as np

from src.dwpose import DWposeDetector
from src.utils.util import get_fps, read_frames, save_videos_from_pil

# Extract dwpose mp4 videos from raw videos
# /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_dwpose/*/*.mp4


def process_single_video_mem(video_path, detector, root_dir, save_dir):
    relative_path = os.path.relpath(video_path, root_dir)
    print(relative_path, video_path, root_dir)
    out_path = os.path.join(save_dir, relative_path)
    if os.path.exists(out_path):
        return

    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # load input stream
    fps = get_fps(video_path)
    container_in = av.open(video_path)
    stream_in = next(s for s in container_in.streams if s.type == "video")
    width, height = stream_in.width, stream_in.height

    # create output stream
    save_fmt = Path(out_path).suffix
    assert save_fmt == ".mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    codec = "libx264"
    container_out = av.open(out_path, "w")
    stream_out = container_out.add_stream(codec, rate=fps)
    stream_out.width = width
    stream_out.height = height

    for packet in container_in.demux(stream_in):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            result, score = detector(image)
            score = np.mean(score, axis=-1)

            av_frame = av.VideoFrame.from_image(result)
            container_out.mux(stream_out.encode(av_frame))

    container_out.mux(stream_out.encode())
    container_out.close()


def process_single_video(video_path, detector, root_dir, save_dir):
    relative_path = os.path.relpath(video_path, root_dir)
    print(relative_path, video_path, root_dir)
    out_path = os.path.join(save_dir, relative_path)
    if os.path.exists(out_path):
        return

    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)
    frames = read_frames(video_path)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        result, score = detector(frame_pil)
        score = np.mean(score, axis=-1)

        kps_results.append(result)

    save_videos_from_pil(kps_results, out_path, fps=fps)


def process_batch_videos(video_list, detector, root_dir, save_dir):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video_mem(video_path, detector, root_dir, save_dir)


if __name__ == "__main__":
    # -----
    # NOTE:
    # python tools/extract_dwpose_from_vid.py --video_root /path/to/video_dir
    # -----
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str)
    parser.add_argument(
        "--save_dir", type=str, help="Path to save extracted pose videos"
    )
    parser.add_argument("-j", type=int, default=4, help="Num workers")
    args = parser.parse_args()
    num_workers = args.j
    if args.save_dir is None:
        save_dir = args.video_root + "_dwpose"
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    print(f"avaliable gpu ids: {gpu_ids}")

    # collect all video_folder paths
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.video_root):
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))
    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)

    # split into chunks,
    batch_size = (len(video_mp4_paths) + num_workers - 1) // num_workers
    print(f"Num videos: {len(video_mp4_paths)} {batch_size = }")
    video_chunks = [
        video_mp4_paths[i : i + batch_size]
        for i in range(0, len(video_mp4_paths), batch_size)
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
                    process_batch_videos, chunk, detector, args.video_root, save_dir
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()

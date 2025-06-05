
import json
import os
import torch
import decord
import argparse

import pandas as pd
import numpy as np
from pprint import pprint
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor

from torch.utils.data import DataLoader

SYSTEM_PROMPT = "I need you to generate a structured and detailed caption for the provided video. The structured output and the requirements for each field are as shown in the following JSON content: {\"subjects\": [{\"appearance\": \"Main subject appearance description\", \"action\": \"Main subject action\", \"expression\": \"Main subject expression  (Only for human/animal categories, empty otherwise)\", \"position\": \"Subject position in the video (Can be relative position to other objects or spatial description)\", \"TYPES\": {\"type\": \"Main category (e.g., Human)\", \"sub_type\": \"Sub-category (e.g., Man)\"}, \"is_main_subject\": true}, {\"appearance\": \"Non-main subject appearance description\", \"action\": \"Non-main subject action\", \"expression\": \"Non-main subject expression (Only for human/animal categories, empty otherwise)\", \"position\": \"Position of non-main subject 1\", \"TYPES\": {\"type\": \"Main category (e.g., Vehicles)\", \"sub_type\": \"Sub-category (e.g., Ship)\"}, \"is_main_subject\": false}], \"shot_type\": \"Shot type(Options: long_shot/full_shot/medium_shot/close_up/extreme_close_up/other)\", \"shot_angle\": \"Camera angle(Options: eye_level/high_angle/low_angle/other)\", \"shot_position\": \"Camera position(Options: front_view/back_view/side_view/over_the_shoulder/overhead_view/point_of_view/aerial_view/overlooking_view/other)\", \"camera_motion\": \"Camera movement description\", \"environment\": \"Video background/environment description\", \"lighting\": \"Lighting information in the video\"}"


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, input_txt, model_path, start_idx=0, end_idx=-1, filter_existing=True):
        self.video_paths = np.loadtxt(input_txt, dtype=str)
        if end_idx == -1:
            end_idx = len(self.video_paths)
            
        if end_idx > len(self.video_paths):
            print(f'end_idx {end_idx} is greater than the number of videos {len(self.video_paths)}')
            end_idx = len(self.video_paths)
            
        self.video_paths = self.video_paths[start_idx:end_idx]
        
        if filter_existing:
            non_existing_paths = []
            for path in self.video_paths:
                json_path = path.replace('/mp4', '/captions').replace('.mp4', '.json')
                if not os.path.exists(json_path):
                    non_existing_paths.append(path)
                else:
                    # check if the json file is valid
                    try:
                        json.load(open(json_path))
                    except Exception as e:
                        non_existing_paths.append(path)
                        print(f'{json_path} is not valid: {e}')
                
            self.video_paths = non_existing_paths
            print(f'{len(non_existing_paths)} videos remain')
                
        print(f'{len(self.video_paths)} videos remain')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
  
    def __getitem__(self, index):        
        path = self.video_paths[index]
        
        vr = decord.VideoReader(path, ctx=decord.cpu(0), width=360, height=420)
        start = 0
        end = len(vr)
        # avg_fps = vr.get_avg_fps()
        index = self.get_index(end-start, 16, st=start)
        frames = vr.get_batch(index).asnumpy() # n h w c
        video_inputs = [torch.from_numpy(frames).permute(0, 3, 1, 2)]
        conversation = {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": path,
                            "max_pixels": 360 * 420, # 460800
                            "fps": 2.0,
                        },
                        {   
                            "type": "text", 
                            "text": SYSTEM_PROMPT
                        },
                    ],
                }
                
        # user_input
        user_input = self.processor.apply_chat_template(
            [conversation],
            tokenize=False,
            add_generation_prompt=True
        )
        results = dict()
        inputs = {
            'prompt': user_input,
            'multi_modal_data': {'video': video_inputs}
        }
        # results["index"] = real_index
        results['input'] = inputs
        results['path'] = path
        return results

    def __len__(self):
        return len(self.video_paths)

    def get_index(self, video_size, num_frames, st=0):
        seg_size = max(0., float(video_size - 1) / num_frames)
        max_frame = int(video_size) - 1
        seq = []
        # index from 1, must add 1
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            # end = int(np.round(seg_size * (i + 1)))
            idx = min(start, max_frame)
            seq.append(idx+st)
        return seq


def worker_init_fn(worker_id):
    # Set different seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Prevent deadlocks by setting timeout
    torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description="SkyCaptioner-V1 vllm batch inference")
    parser.add_argument("--input_txt", default=None, type=str)
    # parser.add_argument("--out_csv", default="./examples/test_result.csv")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model_path", required=True, type=str, help="skycaptioner-v1 model path")
    args = parser.parse_args()
    
    pprint(args)
    
    dataset = VideoTextDataset(args.input_txt, model_path=args.model_path, start_idx=args.start_idx, end_idx=args.end_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        timeout=180,
    )

    sampling_params = SamplingParams(temperature=0.05, max_tokens=2048)
    
    llm = LLM(model=args.model_path,
        gpu_memory_utilization=0.6, 
        max_model_len=31920,
        tensor_parallel_size=args.tp)
    
    for video_batch in tqdm(dataloader):
        # indices = video_batch["index"]
        inputs = video_batch["input"]
        batch_user_inputs = []
        for prompt, video in zip(inputs['prompt'], inputs['multi_modal_data']['video'][0]):
            usi={'prompt':prompt, 'multi_modal_data':{'video':video}}
            batch_user_inputs.append(usi)
        outputs = llm.generate(batch_user_inputs, sampling_params, use_tqdm=False)
        struct_outputs = [output.outputs[0].text for output in outputs]
        for sidx, sout in enumerate(struct_outputs):
            vid_path = video_batch['path'][sidx]
            save_path = vid_path.replace('/mp4', '/captions').replace('.mp4', '.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                with open(save_path, 'w') as f:
                    json.dump(json.loads(sout), f)
            except Exception as e:
                print(f'Error saving to {save_path}: {e}')


if __name__ == '__main__':
    main()
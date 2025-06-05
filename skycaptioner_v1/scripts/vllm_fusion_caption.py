import os
from pathlib import Path
import argparse
import glob
from pprint import pprint
import time
import gc
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import pandas as pd
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
import json
import random
import numpy as np


SYSTEM_PROMPT_I2V = """
You are an expert in video captioning. You are given a structured video caption and you need to compose it to be more natural and fluent in English.

## Structured Input
{structured_input}

## Notes
1. If there has an empty field, just ignore it and do not mention it in the output.
2. Do not make any semantic changes to the original fields. Please be sure to follow the original meaning.

## Output Principles and Orders
1. First, eliminate the static information in the action field that is not related to the timing action, such as background or environment information.
2. Second, describe each subject with its pure action and expression if these fields exist.

## Output
Please directly output the final composed caption without any additional information.
"""

SYSTEM_PROMPT_T2V = """
You are an expert in video captioning. You are given a structured video caption and you need to compose it to be more natural and fluent in English.

## Structured Input
{structured_input}

## Notes
1. According to the action field information, change its name field to the subject pronoun in the action.
2. If there has an empty field, just ignore it and do not mention it in the output.
3. Do not make any semantic changes to the original fields. Please be sure to follow the original meaning.

## Output Principles and Orders
1. First, declare the shot_type, then declare the shot_angle and the shot_position fields.
2. Second, eliminate information in the action field that is not related to the timing action, such as background or environment information if action is not empty.
3. Third, describe each subject with its pure action, appearance, expression, position if these fields exist.
4. Finally, declare the environment and lighting if the environment and lighting fields are not empty.

## Output
Please directly output the final composed caption without any additional information.
"""


SHOT_TYPE_LIST = [
    'close-up shot',
    'extreme close-up shot',
    'medium shot',
    'long shot',
    'full shot',
]


class StructuralCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, input_txt, model_path, task=None, start_idx=0, end_idx=-1, filter_existing=True):
        
        self.video_paths = sorted(np.loadtxt(input_txt, dtype=str))
        if end_idx == -1:
            end_idx = len(self.video_paths)
            
        if end_idx > len(self.video_paths):
            print(f'end_idx {end_idx} is greater than the number of videos {len(self.video_paths)}')
            end_idx = len(self.video_paths)
            
        self.video_paths = self.video_paths[start_idx:end_idx]
        
        if task is None:
            self.task = args.task
        else:
            self.task = task
            
        if filter_existing:
            non_existing_paths = []
            for path in self.video_paths:
                json_path = path.replace('/mp4', '/captions').replace('.mp4', f'_{self.task}.json')
                if not os.path.exists(json_path):
                    non_existing_paths.append(path)
            self.video_paths = non_existing_paths
            print(f'{len(non_existing_paths)} videos remain')
                
        print(f'{len(self.video_paths)} videos remain')

        self.system_prompt = SYSTEM_PROMPT_T2V if self.task == 't2v' else SYSTEM_PROMPT_I2V
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        # row = self.meta.iloc[index]
        # real_index = self.meta.index[index]
        video_path = self.video_paths[index]
        struct_caption_path = video_path.replace('/mp4', '/captions').replace('.mp4', '.json')
        
        struct_caption = json.load(open(struct_caption_path))

        camera_movement = struct_caption.get('camera_motion', '')
        if camera_movement != '':
            camera_movement += '.'
        camera_movement = camera_movement.capitalize()
        
        fusion_by_llm = False
        cleaned_struct_caption = self.clean_struct_caption(struct_caption, self.task)
        if cleaned_struct_caption.get('num_subjects', 0) > 0:
            new_struct_caption = json.dumps(cleaned_struct_caption, indent=4, ensure_ascii=False)
            conversation = [
                {
                    "role": "system",
                    "content": self.system_prompt.format(structured_input=new_struct_caption),
                },
            ]
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            fusion_by_llm = True
        else:
            text = '-'
        return struct_caption_path, fusion_by_llm, text, '-', camera_movement
    
    def clean_struct_caption(self, struct_caption, task):
        raw_subjects = struct_caption.get('subjects', [])
        subjects = []
        for subject in raw_subjects:
            subject_type = subject.get("TYPES", {}).get('type', '')
            subject_sub_type = subject.get("TYPES", {}).get('sub_type', '')
            if subject_type not in ["Human", "Animal"]:
                subject['expression'] = ''
            if subject_type == 'Human' and subject_sub_type == 'Accessory':
                subject['expression'] = ''
            if subject_sub_type != '':
                subject['name'] = subject_sub_type
            if 'TYPES' in subject:
                del subject['TYPES']
            if 'is_main_subject' in subject:
                del subject['is_main_subject']
            
            # remove action and expression for human subjects since we condition with pose
            if subject_type == 'Human':
                subject['action'] = ''
                subject['expression'] = ''
                
            subjects.append(subject)

        to_del_subject_ids = []
        for idx, subject in enumerate(subjects):
            action = subject.get('action', '').strip()
            subject['action'] = action
            if random.random() > 0.9 and 'appearance' in subject:
                del subject['appearance']
            if random.random() > 0.9 and 'position' in subject:
                del subject['position']
            if task == 'i2v':
                # just keep name and action, expression in subjects
                dropped_keys = ['appearance', 'position']
                for key in dropped_keys:
                    if key in subject:
                        del subject[key]
                if subject['action'] == '' and ('expression' not in subject or subject['expression'] == ''):
                    to_del_subject_ids.append(idx)
        
        # delete the subjects according to the to_del_subject_ids
        for idx in sorted(to_del_subject_ids, reverse=True):
            del subjects[idx]


        shot_type = struct_caption.get('shot_type', '').replace('_', ' ')
        # if shot_type not in SHOT_TYPE_LIST:
        #     struct_caption['shot_type'] = ''
        
        new_struct_caption = {
            'num_subjects': len(subjects),
            'subjects': subjects,
            'shot_type': struct_caption.get('shot_type', ''),
            'shot_angle': struct_caption.get('shot_angle', ''),
            'shot_position': struct_caption.get('shot_position', ''),
            'environment': struct_caption.get('environment', ''),
            'lighting': struct_caption.get('lighting', ''),
        }

        if task == 't2v' and random.random() > 0.9:
            del new_struct_caption['lighting']

        if task == 'i2v':
            drop_keys = ['environment', 'lighting', 'shot_type', 'shot_angle', 'shot_position']
            for drop_key in drop_keys:
                del new_struct_caption[drop_key]
        return new_struct_caption

def custom_collate_fn(batch):
    real_indices, fusion_by_llm, texts, original_texts, camera_movements = zip(*batch)
    return list(real_indices), list(fusion_by_llm), list(texts), list(original_texts), list(camera_movements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption Fusion by LLM")
    parser.add_argument("--input_txt", default="./examples/test_result.csv")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model_path", required=True, type=str, help="LLM model path")
    parser.add_argument("--task", default='t2v', help="t2v or i2v", choices=['t2v', 'i2v'])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=512,
        stop=['\n\n']
    )
    # model_path = "/maindata/data/shared/public/Common-Models/Qwen2.5-32B-Instruct/"

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        tensor_parallel_size = args.tp
    )
    

    dataset = StructuralCaptionDataset(
        input_txt=args.input_txt, 
        model_path=args.model_path,
        task=args.task,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=8,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
    )

    for sc_paths, fusion_by_llms, texts, original_texts, camera_movements in tqdm(dataloader):
        sc_paths_, llm_texts, llm_original_texts, llm_camera_movements = [], [], [], []
        for sc_path, fusion_by_llm, text, original_text, camera_movement in zip(sc_paths, fusion_by_llms, texts, original_texts, camera_movements):
            if fusion_by_llm:
                sc_paths_.append(sc_path)
                llm_texts.append(text)
                llm_original_texts.append(original_text)
                llm_camera_movements.append(camera_movement)    
            else:
                caption = original_text + " " + camera_movement
                with open(sc_path.replace('.json', f'_{args.task}.json'), 'w') as f:
                    json.dump(caption, f)
        
        if len(llm_texts) > 0:
            try:
                outputs = llm.generate(llm_texts, sampling_params, use_tqdm=False)
                results = []
                for output in outputs:
                    result = output.outputs[0].text.strip()
                    results.append(result)
            except Exception as e:
                print(f"Error at {sc_paths_}: {str(e)}")
                import ipdb; ipdb.set_trace()
                results = llm_original_texts
            
            for idx, (result, camera_movement) in enumerate(zip(results, llm_camera_movements)):
                # concat camera movement to fusion_caption
                llm_caption = result + " " + camera_movement
                
                with open(sc_paths_[idx].replace('.json', f'_{args.task}.json'), 'w') as f:
                    json.dump(llm_caption, f)
        
        # print(f'Processed {len(sc_paths)} videos')
        # print(f'Saved to:')
        # for sc_path in sc_paths:
        #     print('\t', sc_path.replace('.json', f'_{args.task}.json'))
        # input('Press Enter to continue...')
                
    torch.cuda.empty_cache()
    gc.collect()

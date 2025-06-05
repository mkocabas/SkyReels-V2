import json
import os
import glob
import gradio as gr
import numpy as np
import argparse


def load_video(vid_paths, idx=-1):
    if idx == -1:
        idx = np.random.randint(len(vid_paths))
    vid_path = vid_paths[idx]
    render_video_path = vid_path.replace('/mp4', '/renders').replace('.mp4', '_human_cam.mp4')
    t2v_caption_path = vid_path.replace('/mp4', '/captions').replace('.mp4', '_t2v.json')
    i2v_caption_path = vid_path.replace('/mp4', '/captions').replace('.mp4', '_i2v.json')
    caption_path = vid_path.replace('/mp4', '/captions').replace('.mp4', '.json')
    caption_str = json.dumps(json.load(open(caption_path)), indent=4)
    if os.path.exists(t2v_caption_path):
        t2v_caption_str = json.dumps(json.load(open(t2v_caption_path)), indent=4)
    else:
        t2v_caption_str = 'N/A'
    if os.path.exists(i2v_caption_path):
        i2v_caption_str = json.dumps(json.load(open(i2v_caption_path)), indent=4)
    else:
        i2v_caption_str = 'N/A'
    return vid_path, render_video_path, caption_str, t2v_caption_str, i2v_caption_str


def load_random_video(vid_paths):
    vid_path, render_video_path, caption_str, t2v_caption_str, i2v_caption_str = load_video(vid_paths)
    return vid_path, render_video_path, caption_str, t2v_caption_str, i2v_caption_str, vid_path


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir', type=str, default='/home/data/datasets/BEDLAM/images/*/mp4/*.mp4')
    args = parser.parse_args()

    v1_dir = '/home/data/datasets/BEDLAM/images/*/mp4/*.mp4'
    v2_dir = '/home/data/datasets/BEDLAM2_0/images/*/mp4/*.mp4'
    all_vids = sorted(glob.glob(v1_dir)+ glob.glob(v2_dir))
    
    print(f'Found {len(all_vids)} videos')
    print(f'Example video: {all_vids[0]}')
    
    # Gradio components
    with gr.Blocks() as demo:
        load_button = gr.Button("Load Random Video")
        
        with gr.Row():
            with gr.Column():
                orig_video = gr.Video(label="Original Video", height=480, width=848)
                render_video = gr.Video(label="Rendered Video", height=480, width=848)
            
            with gr.Column():
                t2v_caption = gr.Textbox(label="T2V Caption", lines=7, max_lines=7)
                i2v_caption = gr.Textbox(label="I2V Caption", lines=3, max_lines=3)
                video_path_display = gr.Textbox(label="Video Path", interactive=False)
                caption = gr.Code(label="Caption", language="json", lines=10)

        # Use gr.State to hold the list of video paths
        video_paths_state = gr.State(value=all_vids)

        # Define the button click action
        load_button.click(
            fn=load_random_video,
            inputs=[video_paths_state],
            outputs=[orig_video, render_video, caption, t2v_caption, i2v_caption, video_path_display]
        )

    # Launch the Gradio interface
    demo.launch(share=True, allowed_paths=[args.inp_dir.split('/images')[0]])



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from PIL import Image
import ast
import asyncio
import time
import collections
import argparse

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from sam2.build_sam import build_sam2_camera_predictor
from llm.gpt4o_modeling import GPT4o
from llm.qwen2_modeling import Qwen2
from utils import add_text_with_background


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

async def extract(query, model):
    with open("llm/openie.txt", "r") as file:
        ie_prompt = file.read()
    text = await model.generate(ie_prompt.format_map({"query": query}))
    text = ast.literal_eval(text)["query"]
    
    return text

async def extract_handler(query, queue, model):
    text = await extract(query, model)
    queue.append(text)

def load_model(model):

    # init grounding dino model from huggingface
    # model_id = "IDEA-Research/grounding-dino-tiny"
    model_id = "gdino_checkpoints/grounding-dino-tiny"
    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # build sam2
    sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_configs/sam2_hiera_s.yaml"
    model_cfg = "sam2_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device=device)

    if 'gpt' in model.lower(): # "gpt-4o-2024-05-13"
        gpt4o = GPT4o(model)
        return grounding_processor, grounding_model, predictor, gpt4o
    elif 'qwen' in model.lower(): # "llm_checkpoints/Qwen2-7B-Instruct-AWQ"
        qwen2 = Qwen2(f"llm_checkpoints/{model}", device=device)
        return grounding_processor, grounding_model, predictor, qwen2
    else:
        raise NotImplementedError("INVALID MODEL NAME")



async def main(model="gpt-4o-2024-05-13"):
    
    # load model
    grounding_processor, grounding_model, predictor, llm = load_model(model)
    
    # load video
    # cap = cv2.VideoCapture(0) # camera
    cap = cv2.VideoCapture("notebooks/videos/case.mp4")
    
    # init
    query_queue = collections.deque([])
    response_queue = collections.deque([])
    if_init = False
    frame_list = [] # for visualization
    text = ""
    query = ""
    results = None
    
    idx = 0
    # fps_cut = 2 # skip every fps_cut step to save time
    while True:
        idx += 1    
        print(idx)
        ret, frame = cap.read()
        if not ret:
            break
        # if idx % fps_cut == 0: continue
        
        # simulate query
        if idx == 1:
            query = "I am thirsty"
            query_queue.append(query)
        if idx == 51:
            query = "find a tool for writing."
            query_queue.append(query)

        if query_queue:
            query = query_queue.popleft()
            asyncio.create_task(extract_handler(query, response_queue, llm))
        if response_queue:
            text = response_queue.popleft()
            # print(f"LLM Response: {text}")
            if_init = False

        if text:
            width, height = frame.shape[:2][::-1]    
            if not if_init:
                predictor.load_first_frame(frame)
                

                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
                
                # box from groundingDINO
                # print(frame.shape)
                inputs = grounding_processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), text=text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)
                results = grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.6,
                    text_threshold=0.6,
                    target_sizes=[frame.shape[:2]]
                )
                
                # single box
                boxes = results[0]["boxes"]
                if boxes.shape[0] != 0:
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=boxes,
                    )
                    if_init = True
                else:
                    if_init = False
                
                
                # continue

            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)

                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                # print(all_mask.shape)
                for i in range(0, len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                        np.uint8
                    ) * 255

                    all_mask = cv2.bitwise_or(all_mask, out_mask)

                # print(all_mask.shape, type(all_mask))


                # all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
                all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        
        if query:
            frame = add_text_with_background(frame, query)

            # cv2.imshow("frame", frame)
            # cv2.imwrite(f"output/frame_{idx}.jpg", frame)
            # idx += 1
            
        # Ensure tasks are running
        await asyncio.sleep(0)

        
        frame_list.append(frame)
        # result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    
    # visualization 
    frame_list = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frame_list]
    gif = imageio.mimsave("./result.gif", frame_list, "GIF")
    # w, h = frame_list[0].shape[:2][::-1]
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # # video_handler = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"MP4V"),25,(w,h))
    # # video_handler = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"XVID"),25,(w,h))
    # video_handler = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),int(fps),(w,h))
    # # video_handler = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc(*"XVID"),fps,(w,h))
    # for frame in frame_list:
    #     video_handler.write(frame)
    # video_handler.release()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the system with a specified model.')
    parser.add_argument('--model', type=str, required=True, help='The llm to use for the system.')
    args = parser.parse_args()
    
    asyncio.run(main(args.model))
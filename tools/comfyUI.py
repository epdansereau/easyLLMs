#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from random import randint

import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt, websockets_node_id):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node == websockets_node_id:
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images


def generate(workflow_path, image_prompt):
    with open(workflow_path, "r") as f:
        prompt_text = f.read()

    # The workflows saved by comfyUI can need to be repaired to be loaded as JSON.

    prompt = json.loads(prompt_text)

    # finding the websockets node
    websockets_node_id = None
    for node_id in prompt:
        if prompt[node_id]["_meta"]["title"] == "SaveImageWebsocket":
            websockets_node_id = node_id
            break

    # setting the seed
    for node_id in prompt:
        if prompt[node_id]["class_type"] == "RandomNoise":
            prompt[node_id]["inputs"]["noise_seed"] = randint(0, 1000000000)
            break
        if prompt[node_id]["class_type"] == "HyVideoSampler":
            prompt[node_id]["inputs"]["seed"] = randint(0, 1000000000)
            break

    # setting the prompt text
    for node_id in prompt:
        if prompt[node_id]["class_type"] == "CLIPTextEncode":
            prompt[node_id]["inputs"]["text"] = image_prompt
            break
        if prompt[node_id]["class_type"] == "HyVideoTextEncode":
            prompt[node_id]["inputs"]["prompt"] = image_prompt
            break


    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt, websockets_node_id)
    ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
    #Commented out code to display the output images:

    return images


def test_image_generation():
    workflow_path = "Flux Schnell API.json"
    default_image_prompt = '''a bottle with a beautiful rainbow galaxy inside it on top of a wooden table in the middle of a modern kitchen beside a plate of vegetables and mushrooms and a wine glasse that contains a planet earth with a plate with a half eaten apple pie on it'''

    images = generate(workflow_path, default_image_prompt)
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            image.show()


def test_video_generation():
    workflow_path = "hyvideo_t2v_example_01 API.json"
    default_image_prompt = '''Video of a bottle with a beautiful rainbow galaxy inside it on top of a wooden table in the middle of a modern kitchen beside a plate of vegetables and mushrooms and a wine glasse that contains a planet earth with a plate with a half eaten apple pie on it'''
    images = generate(workflow_path, default_image_prompt)
    frames = []

    # Convert each image from bytes into a NumPy array (OpenCV format)
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frames.append(frame)

    delay = int(1000 / 24)  # delay in milliseconds for 24 FPS

    while True:
        for frame in frames:
            cv2.imshow("Video Playback", frame)
            # If 'q' is pressed at any time, exit the loop
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit(0)

def get_last_tool_call_id(config):
    partial_fn = config['configurable']['__pregel_send']
    
    # Get the PregelExecutableTask object
    task = partial_fn.args[0]

    # Access its .input attribute
    input_dict = task.input

    # Access messages
    messages = input_dict['messages']

    # Grab the last tool_call id
    tool_call_id = messages[-1].tool_calls[0]['id']

    return tool_call_id

@tool
def generate_image(prompt: str, special_config_param: RunnableConfig) -> str:
    """
    Generate an image from a detailed prompt string.

    Args:
        prompt (str): A text prompt describing the image to generate.

    Returns:
        str: A confirmation message indicating the image was successfully generated and displayed.
    """
    tool_call_id = get_last_tool_call_id(special_config_param)

    # Hard-coded workflow path for image generation
    workflow_path = "tools/Flux Schnell API.json"
    
    # Generate image data (assumed to be a dict mapping node IDs to lists of image bytes)
    images_dict = generate(workflow_path, prompt)
    
    # Create directory for generated images if it doesn't exist
    os.makedirs("generated_images", exist_ok=True)
    
    # Save each image to disk with a unique filepath
    for node_id in images_dict:
        for idx, image_data in enumerate(images_dict[node_id]):
            image = Image.open(io.BytesIO(image_data))
            file_path = f"generated_images/image_{tool_call_id}.png"
            image.save(file_path)
            # The file_path is now saved; server-side handling is assumed.
    
    return f"Image successfully generated and displayed to the user."

def generate_video(prompt: str, special_config_param: RunnableConfig) -> str:
    """
    Generate a video 41 frames video from a detailed prompt string.

    Args:
        prompt (str): A text prompt describing the video to generate.

    Returns:
        str: A confirmation message indicating the video was successfully generated and displayed.
    """
    # Hard-coded workflow path for video generation
    tool_call_id = get_last_tool_call_id(special_config_param)

    workflow_path = "tools/hyvideo_t2v_example_01 API.json"
    
    # Generate video frames (assumed to be a dict mapping node IDs to lists of image bytes)
    images_dict = generate(workflow_path, prompt)
    
    # Collect frames in OpenCV BGR format
    frames = []
    for node_id in images_dict:
        for image_data in images_dict[node_id]:
            # Open the image with PIL
            image = Image.open(io.BytesIO(image_data))
            # Convert to OpenCV BGR format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frames.append(frame)
    
    if not frames:
        return "No video frames generated."
    
    # Use the dimensions of the first frame
    height, width, _ = frames[0].shape
    
    # Create directory for the video if it doesn't exist
    os.makedirs("generated_videos", exist_ok=True)
    
    # Define the video filename with a unique timestamp
    video_filepath = f"generated_videos/video_{tool_call_id}.mp4"
    
    # Define video writer with a codec that is typically browser friendly.
    # 'avc1' is a fourcc for H.264, which is widely supported in browsers.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = 24
    video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (width, height))
    
    # Write each frame to the video file, ensuring the frame size matches
    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)
    
    video_writer.release()
    
    return f"Video successfully generated and displayed to the user."
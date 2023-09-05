import os
import activation
import gradio as gr
import subprocess
from PIL import Image
import numpy as np
import shutil

def generate_mesh(input_image):
    input_path = "./input"
    output_path = "./out"
    image_path = input_path + "/input.png"
    GPU_NUM = "0"

    if os.path.exists(input_path):
      shutil.rmtree(input_path)
    if os.path.exists(output_path):
      shutil.rmtree(output_path)

    os.mkdir(input_path)
    input_image.save(image_path)

    cmd_1 = f"python preprocess_image.py --path {image_path}"
    cmd_2 = f"bash scripts/magic123/run_both_priors.sh {GPU_NUM} nerf dmtet {input_path} 1 1"

    try:
        completed_process = subprocess.run(cmd_1.split(), check=True, capture_output=True, text=True)
        completed_process = subprocess.run(cmd_2.split(), check=True, capture_output=True, text=True)
        print(completed_process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.stdout)
        print(e.stderr)

    output_name = f"./out/magic123-nerf-dmtet/magic123_input_nerf_dmtet/mesh/mesh.glb"
    return output_name

#image = Image.open("./0.png")
#generate_mesh(image)

inputs = gr.inputs.Image(label="Image", type="pil")
outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
gr.Interface(generate_mesh, inputs, outputs).launch()

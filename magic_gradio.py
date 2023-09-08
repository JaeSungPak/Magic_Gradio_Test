import os
import activation
import gradio as gr
import subprocess
from PIL import Image
import numpy as np
import shutil
import time
import tqdm
import main_gradio

def reload_package(root_module):
    package_name = root_module.__name__

    # get a reference to each loaded module
    loaded_package_modules = dict([
        (key, value) for key, value in sys.modules.items() 
        if key.startswith(package_name) and isinstance(value, types.ModuleType)])

    # delete references to these loaded modules from sys.modules
    for key in loaded_package_modules:
        del sys.modules[key]

    # load each of the modules again; 
    # make old modules share state with new modules
    for key in loaded_package_modules:
        print 'loading %s' % key
        newmodule = __import__(key)
        oldmodule = loaded_package_modules[key]
        oldmodule.__dict__.clear()
        oldmodule.__dict__.update(newmodule.__dict__)

with gr.Blocks() as demo:
    inputs = gr.inputs.Image(label="Image", type="pil")
    outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
    btn = gr.Button("Generate!")
    
    def generate_mesh(input_image, progress=gr.Progress(track_tqdm=True)):
                
        input_path = "./input"
        output_path = "./out"
        image_path = input_path + "/input.png"
        GPU_NUM = "1"

        if os.path.exists(input_path):
            shutil.rmtree(input_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(input_path)
        input_image.save(image_path)

        cmd_1 = f"python preprocess_image.py --path {image_path}"
        cmd_2 = f"bash scripts/magic123/run_both_priors.sh {GPU_NUM} nerf dmtet {input_path} 1 1"
        try:
            completed_process = subprocess.run(cmd_1.split(), stdout=subprocess.PIPE)
            print(completed_process.stdout)
            
            for i in tqdm.tqdm(range(2), desc="outer"):
                for j in tqdm.tqdm(range(50), desc="inner"):
                    time.sleep(0.1)
                    
            #completed_process = subprocess.run(cmd_2.split(), stdout=subprocess.PIPE)
            main_gradio.run(False)
            reload_package(main_gradio)
            main_gradio.run(True)
            print(completed_process.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(e.stdout)
            print(e.stderr)

        output_name = f"./out/magic123-nerf-dmtet/magic123_input_nerf_dmtet/mesh/mesh.glb"
        return output_name

    btn.click(generate_mesh, inputs, outputs)

#image = Image.open("./0.png")
#generate_mesh(image)

#inputs = gr.inputs.Image(label="Image", type="pil")
#outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
#gr.Interface(generate_mesh, inputs, outputs).launch(share=True)

demo.queue(concurrency_count=20).launch(share=True)

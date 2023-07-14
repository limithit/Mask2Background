import os
import numpy as np
from PIL import Image
import gradio as gr

from datetime import datetime

import cv2
import gc
import logging
from modules.devices import torch_gc
from functools import wraps
from modules import shared, script_callbacks
try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir

ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                          "outputs", "inpaint-anything",
                          datetime.now().strftime("%Y-%m-%d"))

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)

ia_logging = logging.getLogger("Mask2Background")
ia_logging.setLevel(logging.INFO)
ia_logging.propagate = False

ia_logging_sh = logging.StreamHandler()
ia_logging_sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
ia_logging_sh.setLevel(logging.INFO)
ia_logging.addHandler(ia_logging_sh)

def clear_cache():
    gc.collect()
    torch_gc()


def clear_cache_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        clear_cache()
        res = func(*args, **kwargs)
        clear_cache()
        return res
    return wrapper

def update_ia_outputs_dir():
    """Update inpaint-anything outputs directory.
    
    Returns:
        None
    """
    global ia_outputs_dir
    config_save_folder = shared.opts.data.get("inpaint_anything_save_folder", "inpaint-anything")
    if config_save_folder in ["inpaint-anything", "img2img-images"]:
        ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                                      "outputs", config_save_folder,
                                      datetime.now().strftime("%Y-%m-%d"))

@clear_cache_decorator
def input_image_upload(input_image, sam_image, sel_mask):
    global sam_dict
    sam_dict["orig_image"] = input_image
    sam_dict["pad_mask"] = None

    ret_sam_image = np.zeros_like(input_image, dtype=np.uint8) if sam_image is None else gr.update()
    ret_sel_mask = np.zeros_like(input_image, dtype=np.uint8) if sel_mask is None else gr.update()

    return ret_sam_image, ret_sel_mask, gr.update(interactive=True)

@clear_cache_decorator
def create_mask(img, white_bg=True):
    """
    Generate a binary mask of the image (i.e. black and white image)
    Args.
        img_path: str, input image path
        white_bg: bool, if or not change the transparent area to white, default is True
    Returns.
        numpy.ndarray, the binary mask of the image.
    """
    global sam_dict
    black_pixel = (0, 0, 0, 255)
    white_pixel = (255, 255, 255, 255)

    for h in range(img.width):
        for w in range(img.height):
            if img.getpixel((h, w))[3] == 0:
                if white_bg:
                    img.putpixel((h, w), white_pixel)
            else:
                img.putpixel((h, w), black_pixel)
    np_array = np.array(img)
    sam_dict["mask_image"] = np_array
    return np_array

def transparent_to_white(input_image):
    W, H = input_image.size
    white_pixel = (255, 255, 255, 255)
    for h in range(W):
        for w in range(H):
            if input_image.getpixel((h, w))[3] == 0:
                input_image.putpixel((h, w), white_pixel)
    np_array = np.array(input_image)

    return np_array

@clear_cache_decorator
def run_sam(input_image, sam_image):
    # ia_logging.info(f"input_image: {type(input_image)}")
    sam_image = transparent_to_white(input_image)
    width, height = input_image.size
    return gr.update(value=sam_image), f"Fill background with white to complete,Image dimensions: {width}x{height}"

@clear_cache_decorator
def select_mask(input_image, sam_image, invert_chk, sel_mask):
    ret_image = create_mask(input_image)
    return gr.update(value=ret_image)

@clear_cache_decorator
def run_get_mask(input_image):
    ret_image = create_mask(input_image)
    global sam_dict
    if sam_dict["mask_image"] is None:
        return None
    
    mask_image = sam_dict["mask_image"]

    global ia_outputs_dir
    update_ia_outputs_dir()
    if not os.path.isdir(ia_outputs_dir):
        os.makedirs(ia_outputs_dir, exist_ok=True)
    save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
    save_name = os.path.join(ia_outputs_dir, save_name)
    Image.fromarray(mask_image,  mode='RGBA').save(save_name)
    
    return mask_image

def on_ui_tabs():
    global sam_dict
    
    with gr.Blocks(analytics_enabled=False) as Mask2Background_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                        with gr.Row():
                            status_text = gr.Textbox(label="", elem_id="status_text", max_lines=1, show_label=False, interactive=False)
                with gr.Row():
                    input_image = gr.Image(label="Input image", elem_id="input_image", source="upload", type="pil",image_mode="RGBA", interactive=True)
                
                
                with gr.Row():
                    sam_btn = gr.Button("Run Fill the background", elem_id="sam_btn", interactive=False)
                
                with gr.Tab("Mask only", elem_id="mask_only_tab"):
                    with gr.Row():
                        with gr.Column():
                            get_mask_btn = gr.Button("Get mask", elem_id="get_mask_btn")                   
                    with gr.Row():
                        with gr.Column():
                            mask_out_image = gr.Image(label="Get mask image", elem_id="mask_out_image", type="numpy", interactive=False).style(height=480)
                    with gr.Row():
                        with gr.Column():
                            mask_send_to_inpaint_btn = gr.Button("Send to img2img inpaint", elem_id="mask_send_to_inpaint_btn")
            
            with gr.Column():
                with gr.Row():
                    sam_image = gr.Image(label="Fill the background image", elem_id="sam_image", type="numpy", tool="sketch", brush_radius=8,
                                        interactive=True).style(height=480)
                with gr.Row():
                    with gr.Column():
                        select_btn = gr.Button("Create mask", elem_id="select_btn")
                with gr.Row():
                    sel_mask = gr.Image(label="Create mask image", elem_id="sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                        interactive=True).style(height=480)
            
            input_image.upload(input_image_upload, inputs=[input_image, sam_image, sel_mask], outputs=[sam_image, sel_mask, sam_btn])
            sam_btn.click(run_sam, inputs=[input_image, sam_image], outputs=[sam_image, status_text]).then(
                fn=None, inputs=None, outputs=None, _js="Mask2Background_clearSamMask")
            select_btn.click(
                select_mask, 
                inputs=[input_image],
                outputs=[sel_mask])

            get_mask_btn.click(
                run_get_mask,
                inputs=[input_image],
                outputs=[mask_out_image])
            mask_send_to_inpaint_btn.click(
                fn=None,
                _js="Mask2Background_sendToInpaint",
                inputs=None,
                outputs=None)        

    return [(Mask2Background_interface, "Generate Mask&Background", "Mask2Background")]

def on_ui_settings():
    section = ("Mask2Background", "Generate Mask&Background")
    shared.opts.add_option("Mask2Background_save_folder", shared.OptionInfo(
        "Mask2Background", "Folder name where output images will be saved", gr.Radio, {"choices": ["Mask2Background", "img2img-images"]}, section=section))
    shared.opts.add_option("Mask2Background_offline_inpainting", shared.OptionInfo(
        False, "Enable offline network Inpainting", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("Mask2Background_padding_fill", shared.OptionInfo(
        127, "Fill value used when Padding is set to constant", gr.Slider, {"minimum":0, "maximum":255, "step":1}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
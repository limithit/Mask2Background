import os
import numpy as np
from PIL import Image
import gradio as gr

from datetime import datetime

import cv2

from modules import shared, script_callbacks
try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir

from ia_logging import ia_logging

from ia_threading import (clear_cache_decorator)

ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                          "outputs", "inpaint-anything",
                          datetime.now().strftime("%Y-%m-%d"))

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)

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
# 透明区域换成白色，非透明区域换成黑色
# def create_mask(input_image):
#     global sam_dict
#     input_image = input_image.astype(np.uint8)
#     transparent_mask = np.all(input_image == [0, 0, 0], axis=-1)
#     non_transparent_mask = ~transparent_mask

#     white_color = [255, 255, 255]
#     black_color = [0, 0, 0]

#     white_array = np.ones_like(input_image) * white_color
#     black_array = np.ones_like(input_image) * black_color

#     input_image = np.multiply(white_array, transparent_mask[..., np.newaxis]) + np.multiply(black_array, non_transparent_mask[..., np.newaxis])
#     sam_dict["mask_image"] = input_image

#     return input_image
def create_mask(input_image):
    global sam_dict

    # 将输入图像转换为8位无符号整数类型
    input_image_8bit = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 创建掩膜图像
    transparent_mask = np.all(input_image_8bit == [0, 0, 0], axis=-1)
    non_transparent_mask = ~transparent_mask
    white_color = [255, 255, 255]
    black_color = [0, 0, 0]
    white_array = np.ones_like(input_image_8bit) * white_color
    black_array = np.ones_like(input_image_8bit) * black_color
    input_image_8bit = np.multiply(white_array, transparent_mask[..., np.newaxis]) + np.multiply(black_array, non_transparent_mask[..., np.newaxis])

    # 将掩膜图像存储到全局变量sam_dict中
    sam_dict["mask_image"] = input_image_8bit

    return input_image_8bit


# 透明区域换成白色，非透明区域不变
def transparent_to_white(input_image):
    b, g, r = cv2.split(input_image)  # 分解为 RGB 通道

    transparent_mask = (b == 0) & (g == 0) & (r == 0)  # 透明掩码，选中透明像素
    white_color = [255, 255, 255]  # 白色像素颜色
    # white_color = [0, 255, 0]  # 绿色像素颜色，比较明显

    b[transparent_mask] = white_color[0]  # 替换透明像素为白色像素
    g[transparent_mask] = white_color[1]
    r[transparent_mask] = white_color[2]
    return cv2.merge((b, g, r))  # 合并 RGB 通道为图像数组

@clear_cache_decorator
def run_sam(input_image, sam_image):
    ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")
    ia_logging.info(f"input_image: {type(input_image)}")
    sam_image = transparent_to_white(input_image)
    return gr.update(value=sam_image), "Fill background with green to complete"

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
    
    # setup_ia_config_ini()
    # sam_model_index = get_ia_config_index(IAConfig.KEY_SAM_MODEL_ID, IAConfig.SECTION_USER)
    # sam_model_index = sam_model_index if sam_model_index is not None else 1
    # inp_model_index = get_ia_config_index(IAConfig.KEY_INP_MODEL_ID, IAConfig.SECTION_USER)
    # inp_model_index = inp_model_index if inp_model_index is not None else 0
    
    with gr.Blocks(analytics_enabled=False) as inpaint_anything_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                        with gr.Row():
                            status_text = gr.Textbox(label="", elem_id="status_text", max_lines=1, show_label=False, interactive=False)
                with gr.Row():
                    input_image = gr.Image(label="Input image", elem_id="input_image", source="upload", type="numpy", interactive=True)
                
                
                with gr.Row():
                    sam_btn = gr.Button("Run Fill the background", elem_id="sam_btn", interactive=False)
                
                with gr.Tab("Mask only", elem_id="mask_only_tab"):
                    with gr.Row():
                        with gr.Column():
                            get_mask_btn = gr.Button("Get mask", elem_id="get_mask_btn")                   
                    with gr.Row():
                        with gr.Column():
                            mask_out_image = gr.Image(label="Mask image", elem_id="mask_out_image", type="numpy", interactive=False)
                    with gr.Row():
                        # with gr.Column():
                        #     get_alpha_status_text = gr.Textbox(label="", elem_id="get_alpha_status_text", max_lines=1, show_label=False, interactive=False)
                        with gr.Column():
                            mask_send_to_inpaint_btn = gr.Button("Send to img2img inpaint", elem_id="mask_send_to_inpaint_btn")
            
            with gr.Column():
                with gr.Row():
                    sam_image = gr.Image(label="Segment Anything image", elem_id="sam_image", type="numpy", tool="sketch", brush_radius=8,
                                        interactive=True).style(height=480)
                with gr.Row():
                    with gr.Column():
                        select_btn = gr.Button("Create mask", elem_id="select_btn")
                    # with gr.Column():
                    #     invert_chk = gr.Checkbox(label="Invert mask", elem_id="invert_chk", show_label=True, interactive=True)
                with gr.Row():
                    sel_mask = gr.Image(label="Selected mask image", elem_id="sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                        interactive=True).style(height=480)
            
            input_image.upload(input_image_upload, inputs=[input_image, sam_image, sel_mask], outputs=[sam_image, sel_mask, sam_btn])
            sam_btn.click(run_sam, inputs=[input_image, sam_image], outputs=[sam_image, status_text]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSamMask")
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
                _js="inpaintAnything_sendToInpaint",
                inputs=None,
                outputs=None)        

    return [(inpaint_anything_interface, "Generate Mask&Background", "inpaint_anything")]

def on_ui_settings():
    section = ("inpaint_anything", "Generate Mask&Background")
    shared.opts.add_option("inpaint_anything_save_folder", shared.OptionInfo(
        "inpaint-anything", "Folder name where output images will be saved", gr.Radio, {"choices": ["inpaint-anything", "img2img-images"]}, section=section))
    shared.opts.add_option("inpaint_anything_offline_inpainting", shared.OptionInfo(
        False, "Enable offline network Inpainting", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("inpaint_anything_padding_fill", shared.OptionInfo(
        127, "Fill value used when Padding is set to constant", gr.Slider, {"minimum":0, "maximum":255, "step":1}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)

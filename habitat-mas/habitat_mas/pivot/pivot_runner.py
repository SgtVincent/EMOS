"""PIVOT."""

import json
import re
import time
import cv2
import habitat_mas.pivot.pivot as pivot
from PIL import Image
from io import BytesIO
def save_image(image, file_path):
    img = Image.fromarray(image)
    img.save(file_path)
def process_description(gpt_output)->str:
    action_name = gpt_output["action"]
    task_prompt = gpt_output["task_prompt"]
    robot_history = "" if not gpt_output["robot_history"] else f"Robot's history: {gpt_output['robot_history']}"
    return f"""The robot need to {task_prompt}.Now the robot need to {action_name}.{robot_history}"""
    
def make_prompt(description, top_n=3):
    gpt_prompt = process_description(description)
    return f"""
INSTRUCTIONS:
You are tasked to locate an object, region, or point in space in the given annotated image according to a description.
The image is annoated with numbered circles.
Choose the top {top_n} circles that have the most overlap with and/or is closest to what the description is describing in the image.
You are a five-time world champion in this game.
Give a one sentence analysis of why you chose those points.
Provide your answer at the end in a valid JSON of this format:

{{"points": []}}

DESCRIPTION: {gpt_prompt}
IMAGE:
""".strip()


def extract_json(response, key):
    """Extract json data from GPT outputs."""
    json_part = re.search(r"\{.*\}", response, re.DOTALL)
    parsed_json = {}
    if json_part:
        json_data = json_part.group()
        parsed_json = json.loads(json_data)
    else:
        print("No JSON data found ******\n", response)
    return parsed_json[key]


def pivot_perform_selection(prompter, vlm, im, desc, arm_coord, samples, camera_info, top_n):
    """Perform one selection pass given samples."""
    # Plot sampled coords into the input image
    image_circles_np = prompter.add_arrow_overlay_plt(
        image=im, samples=samples, arm_xy=arm_coord, camera_info=camera_info
    )
    #debug for image
    
    time_now = time.time()
    # Encode images and prompts
    # _, encoded_image_circles = cv2.imencode(".png", image_circles_np)
    image_PIL = Image.fromarray(image_circles_np)
    image_buffered = BytesIO()
    image_PIL.save(image_buffered,format = 'PNG')
    prompt_seq = [make_prompt(desc, top_n=top_n), image_buffered]
    response = vlm.query(prompt_seq)
    # Extract infomation from GPT outputs
    try:
        arrow_ids = extract_json(response, "points")
    except Exception as e:
        print(e)
        arrow_ids = []
    return arrow_ids, image_circles_np


def pivot_runner(
    vlm,
    im,
    desc,
    style,
    action_spec,
    actions=None,
    n_samples_init=25,
    n_samples_opt=10,
    n_iters=3,
    n_parallel_trials=1,
    camera_info=None,
):
    """Run pivot."""
    prompter = pivot.VisualIterativePrompter(
        style, action_spec, pivot.SupportedEmbodiments.HF_DEMO
    )
    output_ims = []
    # Initialize center point at the center of the image
    arm_coord = (int(im.shape[1] / 2), int(im.shape[0] / 2))

    for i in range(n_parallel_trials):
        center_mean = action_spec["loc"]
        center_std = action_spec["scale"]
        for itr in range(n_iters):
            if itr == 0:
                # First round, sample n_samples_init points
                style["num_samples"] = n_samples_init
            else:
                # Optimal round, sample n_samples_opt points
                style["num_samples"] = n_samples_opt
            # Sample Actions:
            # 1) random sample actions from 3D world
            # 2) project 3D actions to get 2D coords
            samples = prompter.sample_actions(
                im, arm_coord, center_mean, center_std,
                camera_info, true_action=actions
            )

            # Call GPT to select points:
            # 1) Plot sampled coords into the image
            # 2) Call GPT to select points
            # 3) Extract selected ids and action from the json output
            arrow_ids, image_circles_np = pivot_perform_selection(
                prompter, vlm, im, desc, arm_coord, samples, camera_info, top_n=3
            )

            # Plot selected circles as red
            selected_samples = []
            for sample in samples:
                if int(sample.label) in arrow_ids:
                    sample.coord.color = (255, 0, 0)
                    selected_samples.append(sample)
            # Plot selected points
            image_circles_marked_np = prompter.add_arrow_overlay_plt(
                image_circles_np, selected_samples, arm_coord, camera_info
            )
            output_ims.append(image_circles_marked_np)

            # If at last iteration, pick 1 answer out of the selected ones
            if itr == n_iters - 1:
                # GPT select points and actions
                arrow_ids, _ = pivot_perform_selection(
                    prompter, vlm, im, desc, arm_coord, selected_samples, camera_info, top_n=1
                )

                # Plot selected circles as red
                selected_samples = []
                for sample in samples:
                    if int(sample.label) in arrow_ids:
                        sample.coord.color = (255, 0, 0)
                        selected_samples.append(sample)
                image_circles_marked_np = prompter.add_arrow_overlay_plt(
                    im, selected_samples, arm_coord, camera_info
                )
                output_ims.append(image_circles_marked_np)

            # Fit the Gaussian distribution with the selected points
            center_mean, center_std = prompter.fit(arrow_ids, samples)
    xy_tuple = selected_samples[0].coord.xy
    xy_list = list(xy_tuple)
    return xy_list

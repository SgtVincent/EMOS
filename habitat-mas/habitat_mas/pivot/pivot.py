"""Visual Iterative Prompting functions.

Code to implement visual iterative prompting, an approach for querying VLMs.
"""

import copy
import dataclasses
import enum
import io
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import habitat_mas.pivot.pivot_utils as pivot_utils
from habitat_mas.pivot.pivot_camera_utils import get_coords


@enum.unique
class SupportedEmbodiments(str, enum.Enum):
    HF_DEMO = 'hf_demo'


@dataclasses.dataclass()
class Coordinate:
    # 2D image coordinates for the target annotation
    xy: Tuple[int, int]
    # Color and style of the coord.
    color: Optional[float] = None
    radius: Optional[int] = None


@dataclasses.dataclass()
class Sample:
    # 2D or 3D action
    action: np.ndarray
    # Coordinates for the main annotation
    coord: Coordinate
    # Coordinates for the text label
    text_coord: Coordinate
    # Label to display in the text label
    label: str


class VisualIterativePrompter:
    def __init__(self, style, action_spec, embodiment):
        self.embodiment = embodiment
        self.style = style
        self.action_spec = action_spec
        self.fig_scale_size = None

    def action_to_coord(self, action, image, arm_xy, camera_info, do_project=False):
        """Converts 2D/3D action to 2D image coordinate."""
        # Extract action coords
        action_x = action[0]
        action_z = action[2]
        # If Z-dim is around the similar value
        # actions have high chance to be on the ground: Navigation
        if self.action_spec['scale'][1] <= 0.05:  # ignore z dimension
            action_y = self.action_spec['loc'][1]
        else:
            action_y = action[1]

        coord = get_coords([action_x, action_y, action_z], camera_info)[0]
        image_x, image_y = coord[0], coord[1]

        # Get image size in case the point is out of image boundary
        (original_image_height, original_image_width, _) = image.shape
        image_x = np.clip(image_x, 2 * int(self.style['radius']), original_image_width - 2 * int(self.style['radius']))
        image_y = np.clip(image_y, 2 * int(self.style['radius']), original_image_height - 2 * int(self.style['radius']))

        # Return points as the class Coordinate
        return Coordinate(
            xy=(int(image_x), int(image_y)),
            color=0.1 * self.style['rgb_scale'],
            radius=int(self.style['radius']),
        )

    def sample_actions(
        self, image, arm_xy, loc, scale, camera_info, true_action=None, max_itrs=1000
    ):
        """Randomly Sample 2D/3D actions from the Gaussian distribution"""
        # Initialize variables
        image = copy.deepcopy(image)
        samples = []
        actions = []
        coords = []
        text_coords = []
        labels = []
        true_label = None

        # Keep track of point_cloud action if available.
        if true_action is not None:
            # Randomly choose actions from the point_cloud points
            indices = np.random.choice(
                true_action.shape[0],
                min(len(true_action), int(self.style['num_samples'] / 2)),
                replace=False)
            sampled_actions = true_action[indices]
            # Append sampled points
            for action in sampled_actions:
                actions.append(action)
                coord = self.action_to_coord(action, image, arm_xy, camera_info)
                coords.append(coord)
                text_coords.append(
                    Coordinate(xy=pivot_utils.coord_to_text_coord(coord, arm_xy, coord.radius))
                )
                true_label = np.random.randint(self.style['num_samples'])
                labels.append(str(true_label))

        # Generate all action samples.
        num_samples = max(0, self.style['num_samples'] - len(actions))
        for i in range(num_samples):
            # Ignore points that is sampled from the point cloud points
            if i == true_label:
                continue
            itrs = 0

            # Generate action scaled appropriately.
            action = np.clip(
                np.random.normal(loc, scale),
                self.action_spec['min'],
                self.action_spec['max'],
            )

            # Convert sampled action to image coordinates.
            coord = self.action_to_coord(action, image, arm_xy, camera_info)

            # Resample action if it results in invalid image annotation.
            adjusted_scale = np.array(scale)
            while (
                pivot_utils.is_invalid_coord(coord, coords, self.style['radius'] * 1.5, image)
                or pivot_utils.coord_outside_image(coord, image, self.style['radius'])
            ) and itrs < max_itrs:
                action = np.clip(
                    np.random.normal(loc, adjusted_scale),
                    self.action_spec['min'],
                    self.action_spec['max'],
                )
                coord = self.action_to_coord(action, image, arm_xy, camera_info)
                itrs += 1
                # increase sampling range slightly if not finding a good sample
                adjusted_scale = np.float64(adjusted_scale)*1.1
                if itrs == max_itrs:
                    # If the final iteration results in invalid annotation, just clip to edge of image.
                    coord = self.action_to_coord(action, image, arm_xy, camera_info, do_project=True)

            # Compute image coordinates of text labels.
            radius = coord.radius
            text_coord = Coordinate(
                xy=pivot_utils.coord_to_text_coord(coord, arm_xy, radius)
            )

            actions.append(action)
            coords.append(coord)
            text_coords.append(text_coord)
            labels.append(str(i))

        for i in range(len(actions)):
            sample = Sample(
                action=actions[i],
                coord=coords[i],
                text_coord=text_coords[i],
                label=str(i),
            )
            samples.append(sample)
        return samples

    def add_arrow_overlay_plt(self, image, samples, arm_xy, camera_info):
        """Plot coords in the image"""
        # Add transparent arrows and circles
        overlay = image.copy()
        (original_image_height, original_image_width, _) = image.shape

        white = (
            self.style['rgb_scale'],
            self.style['rgb_scale'],
            self.style['rgb_scale'],
        )
        new_samples = []
        # update samples
        for sample in samples:
            coord = self.action_to_coord(sample.action, image, arm_xy, camera_info)
            x = np.clip(coord.xy[0], 2 * coord.radius, original_image_width - 2 * coord.radius)
            y = np.clip(coord.xy[1], 2 * coord.radius, original_image_height - 2 * coord.radius)
            sample.coord.xy = (x, y)

            radius = sample.coord.radius
            sample.text_coord = Coordinate(
                xy=pivot_utils.coord_to_text_coord(sample.coord, arm_xy, radius)
            )
            new_samples.append(sample)

        # Add arrows.
        for sample in new_samples:
            color = sample.coord.color
            cv2.arrowedLine(
                overlay, arm_xy, sample.coord.xy, color,
                self.style['thickness']
            )
        image = cv2.addWeighted(
            overlay,
            self.style['arrow_alpha'],
            image,
            1 - self.style['arrow_alpha'],
            0,
        )

        overlay = image.copy()
        # Add circles.
        for sample in new_samples:
            color = sample.coord.color
            radius = sample.coord.radius
            cv2.circle(
                overlay,
                sample.text_coord.xy,
                radius,
                color,
                self.style['thickness'] + 1,
            )
            cv2.circle(overlay, sample.text_coord.xy, radius, white, -1)
        image = cv2.addWeighted(
            overlay,
            self.style['circle_alpha'],
            image,
            1 - self.style['circle_alpha'],
            0,
        )

        dpi = plt.rcParams['figure.dpi']
        if self.fig_scale_size is None:
            # test saving a figure to decide size for text figure
            fig_size = (
            original_image_width / dpi, original_image_height / dpi)
            plt.subplots(1, figsize=fig_size)
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            fig = plt.gcf()
            fig.tight_layout(pad=0)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            test_image = cv2.imdecode(
                np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.fig_scale_size = original_image_width / test_image.shape[1]

        # Add text to figure.
        fig_size = (
            self.fig_scale_size * original_image_width / dpi,
            self.fig_scale_size * original_image_height / dpi,
        )
        plt.subplots(1, figsize=fig_size)
        plt.imshow(image, cmap='binary')
        for sample in new_samples:
            plt.text(
                sample.text_coord.xy[0],
                sample.text_coord.xy[1],
                sample.label,
                ha='center',
                va='center',
                color='k',
                fontsize=self.style['fontsize'],
            )

        # Compile image.
        plt.axis('off')
        fig = plt.gcf()
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        image = cv2.imdecode(
            np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
        )

        image = cv2.resize(image, (original_image_width, original_image_height))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return image

    def fit(self, values, samples):
        """Fit a loc and scale to selected actions."""
        actions = [sample.action for sample in samples]
        labels = [sample.label for sample in samples]
        str_values = [str(num) for num in values]
        if not str_values or not bool(set(str_values) & set(labels)):
            # If no response, revert to initial distribution
            print('GPT failed to return integer arrows')
            loc = self.action_spec['loc']
            scale = self.action_spec['scale']
        elif len(str_values) == 1:
            # If single response, add a distribution over it with min scale
            index = np.where([label == str(str_values[-1]) for label in labels])[0][0]
            action = actions[index]
            loc = action
            scale = self.action_spec['min_scale']
        else:
            # Expected situation, fit distribution
            selected_actions = []
            for value in str_values:
                idx = np.where([label == str(value) for label in labels])[0][0]
                selected_actions.append(actions[idx])
            loc_scale = [
                scipy.stats.norm.fit([action[d] for action in selected_actions])
                for d in range(3)
            ]
            loc = [loc_scale[d][0] for d in range(3)]
            scale = np.clip(
                [loc_scale[d][1] for d in range(3)],
                self.action_spec['min_scale'],
                None,
            )

        return loc, scale

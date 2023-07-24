"""Tool for generating images."""
import json
import logging

from langchain.agents import Tool
from steamship import Steamship
from steamship.base.error import SteamshipError

NAME = "GenerateImage"

DESCRIPTION = """
Useful for when you need to generate an image. 
Input: A detailed prompt describing an image. Add SINGLE CHARACTER to your prompt if you want to be featured alone in the picture, like in a selfie.
Output: the UUID of a generated image
"""

PLUGIN_HANDLE = "stable-diffusion"

NEGATIVE_PROMPT = (
    "(nsfw:1.4),easynegative,(deformed, distorted,disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb,"
    " missing limb, (mutated hands and finger:1.4), disconnected limbs, mutation, mutated, ugly, "
    "disgusting, blurry, amputation"
)

PROMPT_TEMPLATE = (
    "Full body portrait, Hyper realistic Rick Sanchez SINGLE CHARACTER,"
    "strong features, neon space glow in the dark colors, "
    "hyper detailed, shot on Hasselblad H4D 200MS Digital Camera, Mitakon Speedmaster 65mm f/1. 4 XCD, "
    "Galaxy, Dimensional effect, Fresnel lighting --q 5 --ar 1:2 --s 1000"
)


class GenerateImageTool(Tool):
    """Tool used to generate images from a text-prompt."""

    client: Steamship

    def __init__(self, client: Steamship):
        super().__init__(
            name=NAME,
            func=self.run,
            description=DESCRIPTION,
            client=client,
        )

    @property
    def is_single_input(self) -> bool:
        """Whether the tool only accepts a single input."""
        return True

    def run(self, prompt: str, **kwargs) -> str:
        """Respond to LLM prompt."""

        # Use the Steamship DALL-E plugin.
        image_generator = self.client.use_plugin(
            plugin_handle=PLUGIN_HANDLE, config={"n": 1, "size": "768x768"}
        )

        logging.info(f"[{self.name}] {prompt}")
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt)

        task = image_generator.generate(text=prompt + PROMPT_TEMPLATE,
                                        options={"negative_prompt": NEGATIVE_PROMPT},
                                        append_output_to_file=True)
        task.wait()
        blocks = task.output.blocks
        logging.info(f"[{self.name}] got back {len(blocks)} blocks")
        if len(blocks) > 0:
            logging.info(f"[{self.name}] image size: {len(blocks[0].raw())}")
            return blocks[0].id
        raise SteamshipError(f"[{self.name}] Tool unable to generate image!")

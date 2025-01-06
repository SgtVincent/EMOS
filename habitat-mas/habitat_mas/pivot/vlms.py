"""VLM Helper Functions."""
import base64
import numpy as np
from openai import OpenAI
import openai
import cv2
import time
from io import BytesIO
class GPT4V:
    """GPT4V VLM."""

    def __init__(self, openai_api_key,openai_base_url):
        # self.client = OpenAI(api_key=openai_api_key)
        openai.api_key = openai_api_key
        openai.base_url = openai_base_url

    def query(self, prompt_seq, temperature=0, max_tokens=512):
        """Queries GPT-4V."""
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                content.append({'type': 'text', 'text': elem})
            elif isinstance(elem, BytesIO):
                base64_image_str = base64.b64encode(elem.getvalue()).decode('utf-8')
                image_url = f'data:image/jpeg;base64,{base64_image_str}'
                content.append(
                    {'type': 'image_url', 'image_url': {'url': image_url}})

        messages = [{'role': 'user', 'content': content}]

        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

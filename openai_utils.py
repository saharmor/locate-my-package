import json
import os
from openai import OpenAI

from image_utils import encode_image

SYSTEM_PROMPT = """
Take a deep breath and work on this problem step by step.
Help me find my delivered packages. Read as many package labels as possible in the attached image. Your output should be a JSON in the following format:
{
id: 1, #sequence number, increment by one for each new package
box_location: "top self, right", # description of where the box is so I can find it
box_label: "John Arnold, Apt #200", # read as much as you can from the package's label
}

"""

system_prompt_img = encode_image('system_img.jpg')


def generate_new_line(base64_image, transcription: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": transcription},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def process_img(base64_image, transcription: str, conversation: list = []):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{system_prompt_img}",
                    },
                ],
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{system_prompt_img}",
                    },
                ],
            },
        ]
        + conversation
        + generate_new_line(base64_image, transcription),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    json_content = response_text.strip('```json\n').rstrip('\n```')
    return json.loads(json_content)


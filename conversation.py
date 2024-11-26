import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import random
IGNORE_INDEX=-100
special_tokens = {
    "vector_start": "<v_start>",
    "vector_end": "<v_end>",
    "vector_patch": "<v_patch>",
    "vector": "<vector>"
    # "class_start": "<class_start>",
    # "class_end": "<class_end>"
}
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str

    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message  + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message   + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message  + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((224, 224))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    # image = image.resize((224, 224))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace('<image>', img_str)
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,

            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }
#######system
conv_chronosight_v1 = Conversation(
    system="You are ChronoSight, a large-scale language and vector classification assistant trained by Radar."
            "Given some vector-class pairs, you are capable of understanding the relationships between them and classifying a new vector into one of the classes."
            "Please pay attention to the underlying relationships between the vector and the class.",
    # tail="Your answer template: Based on the given Vector - Class pairs, the predicted Class for the Vector is Class:",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_chronosight_v1
conv_templates = {
    "default": conv_v1_2,
    "simple": simple_conv,
    "simple_legacy": simple_conv_legacy,
    "multimodal": simple_conv_multimodal,
    "mpt_multimodal": simple_conv_mpt_multimodal,
    "llava_v1": conv_llava_v1, 
    "graphchat_v1": conv_graphchat_v1, 
    "chronosight_v1":conv_chronosight_v1,

    # fastchat
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
    "vicuna_v1_1": conv_vicuna_v1_1,
    "mpt": conv_mpt,
    "mpt_text": conv_mpt_text,
}

class PromptGeneratorSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PromptGeneratorSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self, special_tokens):
        if not hasattr(self, 'initialized'):  # 确保只初始化一次
            self.special_tokens = special_tokens
            self.templates = {
                "template_1": self.template_1,
                "template_2": self.template_2,
                "template_3": self.template_3,
                "template_4": self.template_4,
                "template_5": self.template_5
            }
            self.initialized = True

    def template_1(self, classes):
        human_prompt = ("I need your help in classifying a new vector based on a few examples. "
                "Here are some Vector-Class pairs for your reference:\n")
        for c in classes[:-1]:
            human_prompt += (f"Vector: {special_tokens['vector']}-Class: {c}, ")

        unique_eles = list(set(classes))
        human_prompt += (f"\nThere are {len(unique_eles)} unique Classes, namely: ")
        for i, ele in enumerate(unique_eles):
            human_prompt += (f"{ele} (Class {i+1}), ")

        human_prompt += (f"\nNow, given a new Vector: {special_tokens['vector']}, "
                        "can you determine which Class it belongs to? Your response should look like this:\n"
                        "Based on the given examples, the predicted Class for the Vector is Class:\n")
        response = (f"Based on the given examples, the predicted Class for the Vector "
                    f"is Class: {classes[-1]}.\n")
        return human_prompt, response
    def template_2(self, classes):
        human_prompt = ("Your task is to classify a new vector using a few-shot learning approach. "
                        "Below are some Vector-Class examples to guide you:\n")
        for c in classes[:-1]:
            human_prompt += (f"Example: Vector {special_tokens['vector']} belongs to Class: {c}.\n")

        unique_eles = list(set(classes))
        human_prompt += (f"\nThe known Classes in this task are: ")
        for i, ele in enumerate(unique_eles):
            human_prompt += (f"Class {i+1}: {ele}; ")

        human_prompt += (f"\nNow, analyze the new Vector: {special_tokens['vector']} "
                        "and predict its Class. Please structure your answer as follows:\n"
                        "The predicted Class for the Vector is Class:\n")
        response = (f"The predicted Class for the Vector is Class: {classes[-1]}.\n")
        return human_prompt, response
    def template_3(self, classes):
        human_prompt = ("We are training a classification model. As part of the training, "
                        "we provide the following Vector-Class pairs:\n")
        for c in classes[:-1]:
            human_prompt += (f"Vector: {special_tokens['vector']} => Class: {c}; ")

        unique_eles = list(set(classes))
        human_prompt += (f"\nIn this task, there are {len(unique_eles)} possible Classes: ")
        for i, ele in enumerate(unique_eles):
            human_prompt += (f"{ele} (Class {i+1}), ")

        human_prompt += (f"\nNow, predict the Class for the new Vector: {special_tokens['vector']}.\n"
                        "Answer format:\n"
                        "Predicted Class: Class:\n")
        response = (f"Predicted Class: Class: {classes[-1]}.\n")
        return human_prompt, response
    def template_4(self, classes):
        human_prompt = ("Let's solve a classification problem step by step. First, observe the following "
                        "examples of Vector-Class mappings:\n")
        for c in classes[:-1]:
            human_prompt += (f"- Vector: {special_tokens['vector']}, Class: {c}\n")

        unique_eles = list(set(classes))
        human_prompt += (f"\nWe have identified {len(unique_eles)} unique Classes: ")
        for i, ele in enumerate(unique_eles):
            human_prompt += (f"{ele} (Class {i+1}), ")

        human_prompt += (f"\nNow, apply logical reasoning to determine the Class of a new Vector: {special_tokens['vector']}.\n"
                        "State your prediction as follows:\n"
                        "Conclusion: The Class for the Vector is Class:\n")
        response = (f"Conclusion: The Class for the Vector is Class: {classes[-1]}.\n")
        return human_prompt, response
    
    def template_4(self, classes):
        human_prompt = ("You are an expert classifier. Given the following training examples:\n")
        for c in classes[:-1]:
            human_prompt += (f"- Training example: Vector: {special_tokens['vector']} belongs to Class: {c}\n")

        unique_eles = list(set(classes))
        human_prompt += (f"\nThe problem involves {len(unique_eles)} distinct Classes: ")
        for i, ele in enumerate(unique_eles):
            human_prompt += (f"Class {i+1}: {ele}; ")

        human_prompt += (f"\nNow, as an expert, predict the Class of the following new Vector: {special_tokens['vector']}.\n"
                        "Provide your answer in this format:\n"
                        "The predicted Class is Class:\n")
        response = (f"The predicted Class is Class: {classes[-1]}.\n")
        return human_prompt, response
    
    def _format_response(self, idx, human_prompt, response):
        """
        Helper method to format the response into the desired output structure.
        """
        human_prompt_dict = {'from': 'human', 'value': human_prompt}
        gpt_prompt_dict = {'from': 'gpt', 'value': response}
        sources = {
            'id': idx,
            'conversations': [human_prompt_dict, gpt_prompt_dict]
        }
        return sources
    def generate_prompt(self, idx, classes):
        chosen_template = random.choice(list(self.templates.values()))
        human_prompt, response=  chosen_template(classes)
        return self._format_response(idx, human_prompt, response)


if __name__ == "__main__":
    print(default_conversation.get_prompt())

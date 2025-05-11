import spaces
import gradio as gr
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, LlavaForConditionalGeneration, TextIteratorStreamer
import torch
import torch.amp.autocast_mode
from PIL import Image
import torchvision.transforms.functional as TVF
from threading import Thread
from typing import Generator


MODEL_PATH = "fancyfeast/llama-joycaption-beta-one-hf-llava"
TITLE = "<h1><center>JoyCaption Beta One - (2025-05-10a)</center></h1>"
DESCRIPTION = """
<div>
<p></p>
<p>**This model cannot see any chat history.**</p>
<p>🚨🚨🚨 If the "Help improve JoyCaption" box is checked, the _text_ query you write will be logged and I _might_ use it to help improve JoyCaption.
It does not log images, user data, etc; only the text query.  I cannot see what images you send, and frankly, I don't want to.  But knowing what kinds of instructions
and queries users want JoyCaption to handle will help guide me in building JoyCaption's dataset.  This dataset will be made public.  As always, the model itself is completely
public and free to use outside of this space.  And, of course, I have no control nor access to what HuggingFace, which are graciously hosting this space, collects.</p>
</div>
"""

PLACEHOLDER = """
"""



# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Expected PreTrainedTokenizer, got {type(tokenizer)}"

model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype="bfloat16", device_map=0)
assert isinstance(model, LlavaForConditionalGeneration), f"Expected LlavaForConditionalGeneration, got {type(model)}"


def trim_off_prompt(input_ids: list[int], eoh_id: int, eot_id: int) -> list[int]:
	# Trim off the prompt
	while True:
		try:
			i = input_ids.index(eoh_id)
		except ValueError:
			break
		
		input_ids = input_ids[i + 1:]
	
	# Trim off the end
	try:
		i = input_ids.index(eot_id)
	except ValueError:
		return input_ids
	
	return input_ids[:i]

end_of_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
end_of_turn_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)


@spaces.GPU()
@torch.no_grad()
def chat_joycaption(message: dict, history, temperature: float, top_p: float, max_new_tokens: int, log_prompt: bool) -> Generator[str, None, None]:
	torch.cuda.empty_cache()

	chat_interface.chatbot_state

	# Prompts are always stripped in training for now
	prompt = message['text'].strip()

	# Load image
	if "files" not in message or len(message["files"]) != 1:
		yield "ERROR: This model requires exactly one image as input."
		return
	
	image = Image.open(message["files"][0])
	
	# Log the prompt
	if log_prompt:
		print(f"Prompt: {prompt}")

	# Preprocess image
	# NOTE: I found the default processor for so400M to have worse results than just using PIL directly
	if image.size != (384, 384):
		image = image.resize((384, 384), Image.LANCZOS)
	image = image.convert("RGB")
	pixel_values = TVF.pil_to_tensor(image)

	convo = [
		{
			"role": "system",
			"content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
		},
		{
			"role": "user",
			"content": prompt,
		},
	]

	# Format the conversation
	convo_string = tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
	assert isinstance(convo_string, str)

	# Tokenize the conversation
	convo_tokens = tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

	# Repeat the image tokens
	input_tokens = []
	for token in convo_tokens:
		if token == model.config.image_token_index:
			input_tokens.extend([model.config.image_token_index] * model.config.image_seq_length)
		else:
			input_tokens.append(token)
	
	input_ids = torch.tensor(input_tokens, dtype=torch.long)
	attention_mask = torch.ones_like(input_ids)

	# Move to GPU
	input_ids = input_ids.unsqueeze(0).to("cuda")
	attention_mask = attention_mask.unsqueeze(0).to("cuda")
	pixel_values = pixel_values.unsqueeze(0).to("cuda")

	# Normalize the image
	pixel_values = pixel_values / 255.0
	pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
	pixel_values = pixel_values.to(torch.bfloat16)

	streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

	generate_kwargs = dict(
		input_ids=input_ids,
		pixel_values=pixel_values,
		attention_mask=attention_mask,
		max_new_tokens=max_new_tokens,
		do_sample=True,
		suppress_tokens=None,
		use_cache=True,
		temperature=temperature,
		top_k=None,
		top_p=top_p,
		streamer=streamer,
	)

	if temperature == 0:
		generate_kwargs["do_sample"] = False
	
	t = Thread(target=model.generate, kwargs=generate_kwargs)
	t.start()

	outputs = []
	for text in streamer:
		outputs.append(text)
		yield "".join(outputs)


chatbot=gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface', type="messages")
textbox = gr.MultimodalTextbox(file_types=["image"], file_count="single")

with gr.Blocks() as demo:
	gr.HTML(TITLE)
	chat_interface = gr.ChatInterface(
		fn=chat_joycaption,
		chatbot=chatbot,
		type="messages",
		fill_height=True,
		multimodal=True,
		textbox=textbox,
		additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=True, render=False),
		additional_inputs=[
			gr.Slider(minimum=0,
						maximum=1, 
						step=0.1,
						value=0.6, 
						label="Temperature", 
						render=False),
			gr.Slider(minimum=0,
			 			maximum=1,
						step=0.05,
						value=0.9,
						label="Top p",
						render=False),
			gr.Slider(minimum=8, 
						maximum=4096,
						step=1,
						value=1024, 
						label="Max new tokens", 
						render=False ),
			gr.Checkbox(label="Help improve JoyCaption by logging your text query", value=True, render=False),
		],
    )
	gr.Markdown(DESCRIPTION)


if __name__ == "__main__":
    demo.launch()
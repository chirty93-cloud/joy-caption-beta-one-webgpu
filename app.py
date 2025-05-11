import spaces
import gradio as gr
from transformers import LlavaForConditionalGeneration, TextIteratorStreamer, AutoProcessor
import torch
from PIL import Image
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
CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a descriptive caption for this image in a formal tone.",
		"Write a descriptive caption for this image in a formal tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a formal tone.",
	],
	"Descriptive (Informal)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Training Prompt": [
		"Write a stable diffusion prompt for this image.",
		"Write a stable diffusion prompt for this image within {word_count} words.",
		"Write a {length} stable diffusion prompt for this image.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Booru tag list": [
		"Write a list of Booru tags for this image.",
		"Write a list of Booru tags for this image within {word_count} words.",
		"Write a {length} list of Booru tags for this image.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}



# Load model
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype="bfloat16", device_map=0)
assert isinstance(model, LlavaForConditionalGeneration), f"Expected LlavaForConditionalGeneration, got {type(model)}"
model.eval()


def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
	# Choose the right template row in CAPTION_TYPE_MAP
	if caption_length == "any":
		map_idx = 0
	elif isinstance(caption_length, str) and caption_length.isdigit():
		map_idx = 1  # numeric-word-count template
	else:
		map_idx = 2  # length descriptor template
	
	prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

	if extra_options:
		prompt += " " + " ".join(extra_options)
	
	return prompt.format(
		name=name_input or "{NAME}",
		length=caption_length,
		word_count=caption_length,
	)


@spaces.GPU()
@torch.no_grad()
def chat_joycaption(input_image: Image.Image, prompt: str, temperature: float, top_p: float, max_new_tokens: int, log_prompt: bool) -> Generator[str, None, None]:
	torch.cuda.empty_cache()

	if log_prompt:
		print(f"PromptLog: {repr(prompt)}")

	convo = [
		{
			"role": "system",
			# Beta One supports a wider range of system prompts, but this is a good default
			"content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
		},
		{
			"role": "user",
			"content": prompt.strip(),
		},
	]

	# Format the conversation
	# WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
	# but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
	# if not careful, which can make the model perform poorly.
	convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
	assert isinstance(convo_string, str)

	# Process the inputs
	inputs = processor(text=[convo_string], images=[input_image], return_tensors="pt").to('cuda')
	inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

	streamer = TextIteratorStreamer(processor.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

	generate_kwargs = dict(
		**inputs,
		max_new_tokens=max_new_tokens,
		do_sample=True if temperature > 0 else False,
		suppress_tokens=None,
		use_cache=True,
		temperature=temperature,
		top_k=None,
		top_p=top_p,
		streamer=streamer,
	)

	t = Thread(target=model.generate, kwargs=generate_kwargs)
	t.start()

	outputs = []
	for text in streamer:
		outputs.append(text)
		yield "".join(outputs)


with gr.Blocks() as demo:
	gr.HTML(TITLE)

	with gr.Row():
		with gr.Column():
			input_image = gr.Image(type="pil", label="Input Image")

			caption_type = gr.Dropdown(
				choices=list(CAPTION_TYPE_MAP.keys()),
				value="Descriptive",
				label="Caption Type",
			)

			caption_length = gr.Dropdown(
				choices=["any", "very short", "short", "medium-length", "long", "very long"] +
						[str(i) for i in range(20, 261, 10)],
				label="Caption Length",
				value="long",
			)

			extra_options = gr.CheckboxGroup(
				choices=[
					"If there is a person/character in the image you must refer to them as {name}.",
					"Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
					"Include information about lighting.",
					"Include information about camera angle.",
					"Include information about whether there is a watermark or not.",
					"Include information about whether there are JPEG artifacts or not.",
					"If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
					"Do NOT include anything sexual; keep it PG.",
					"Do NOT mention the image's resolution.",
					"You MUST include information about the subjective aesthetic quality of the image from low to very high.",
					"Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
					"Do NOT mention any text that is in the image.",
					"Specify the depth of field and whether the background is in focus or blurred.",
					"If applicable, mention the likely use of artificial or natural lighting sources.",
					"Do NOT use any ambiguous language.",
					"Include whether the image is sfw, suggestive, or nsfw.",
					"ONLY describe the most important elements of the image."
				],
				label="Extra Options"
			)
			
			name_input = gr.Textbox(label="Person / Character Name")

			with gr.Accordion("Generation settings", open=False):
				temperature_slider = gr.Slider(
					minimum=0.0, maximum=2.0, value=0.6, step=0.05,
					label="Temperature",
					info="Higher values make the output more random, lower values make it more deterministic."
				)
				top_p_slider = gr.Slider(
					minimum=0.0, maximum=1.0, value=0.9, step=0.01,
					label="Top-p"
				)
				max_tokens_slider = gr.Slider(
					minimum=1, maximum=2048, value=512, step=1,
					label="Max New Tokens",
					info="Maximum number of tokens to generate.  The model will stop generating if it reaches this limit."
				)
			
			log_prompt = gr.Checkbox(value=True, label="Help improve JoyCaption by logging your text query")
		
		with gr.Column():
			prompt_box = gr.Textbox(lines=4, label="Prompt", interactive=True)

			# Auto-update prompt box whenever any of the inputs change
			for ctrl in (caption_type, caption_length, extra_options, name_input):
				ctrl.change(
					build_prompt,
					inputs=[caption_type, caption_length, extra_options, name_input],
					outputs=prompt_box,
				)
			
			run_button = gr.Button("Caption")

			output_caption = gr.Textbox(label="Caption")
	
	run_button.click(
		chat_joycaption,
		inputs=[input_image, prompt_box, temperature_slider, top_p_slider, max_tokens_slider, log_prompt],
		outputs=output_caption,
	)

	gr.Markdown(DESCRIPTION)


if __name__ == "__main__":
    demo.launch()
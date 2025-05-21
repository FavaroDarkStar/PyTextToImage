from diffusers import StableDiffusion3Pipeline
import torch,os
from transformers import CLIPTokenizer

#pip install protobuf, sentencepiece
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("❌ Token da Hugging Face não encontrado. Defina HF_TOKEN no arquivo .env.")

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16,
    token=HF_TOKEN
)
pipe.to("cuda")



print("Gerando prompt")
ini_prompt = "Design a logo for a spiritual and self-knowledge brand named @tarot.bymile.The brand is focused on Tarot, spiritual development, and natural magic, aiming to be warm, ethical, inviting, and lightly mystical.The logo should combine elements of elegance, minimalism, and subtle mysticism, avoiding cliché spiritual icons.Preferred visual style: clean, vector-based, professional for digital use.Primary color: #993d74 (rosy magenta). Neutral backgrounds or soft contrasts are welcome"
describe = "Combination mark: balanced layout of text with symbolic visual, suitable for social media and website use."
end_prompt = "Output should look clean and scalable — appropriate for Instagram profile pictures, social banners, and websites.The brand tone is welcoming, thoughtful, and quietly magical, with a grounded, ethical presence."



prompt = ini_prompt+describe+end_prompt

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokens = tokenizer(prompt, return_tensors="pt")
if len(tokens['input_ids'][0]) >= 77:
    exit(print(f"Número de tokens: {len(tokens['input_ids'][0])}"))


print(f"Prompt: {prompt}")
print("Gerando imagem...")
image = pipe(prompt).images[0]
print("Salvando imagem")
image.save("Stable-Diffusion-3-Medium-Diffusers.png")
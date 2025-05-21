from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import json
from datetime import datetime
from PIL import Image
from pathlib import Path
import time
from diffusers.utils import logging as diffusers_logging
import logging
from transformers import logging as transformers_logging
from dotenv import load_dotenv
import gc
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("❌ Token da Hugging Face não encontrado. Defina HF_TOKEN no arquivo .env.")

# Diretórios
DIR_REFERENCIAS = Path("imagens_ref")
DIR_RESULTADOS = Path("imagens_result")
DIR_RESULTADOS.mkdir(exist_ok=True)

def obter_nome_prompt_unico(base_nome="prompt"):
    existente = [p.name for p in DIR_RESULTADOS.iterdir() if p.is_dir() and p.name.startswith(base_nome)]
    contador = 1
    while f"{base_nome}_{contador}" in existente:
        contador += 1
    return f"{base_nome}_{contador}"

def tratar_imagens_referencia(verbose=True):
    tratados = []
    start_total = time.time()

    formatos_suportados = ("*.jpg", "*.jpeg", "*.png")
    arquivos = []
    for ext in formatos_suportados:
        arquivos.extend(DIR_REFERENCIAS.glob(ext))

    for img_path in arquivos:
        if "_tratada" not in img_path.stem:    
            if verbose:
                print(f"→ Tratando imagem de referência: {img_path.name}")
            start = time.time()

            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((512, 512))
                tratado_path = DIR_REFERENCIAS / f"{img_path.stem}_tratada.jpeg"
                img.save(tratado_path)
                tratados.append(str(tratado_path))

            img_path.unlink()  # Remove original

            if verbose:
                print(f"   ✔ Imagem tratada em {time.time() - start:.2f}s")
        else:
            tratados.append(str(img_path))

    if verbose:
        print(f"✅ Tratamento finalizado. Total: {len(tratados)} imagens. Tempo total: {time.time() - start_total:.2f}s\n")

    return tratados


def carrega_modelo(verbose=True):
    if verbose:
        print("⚙️  Carregando modelo...")
    start = time.time()
    
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        token=HF_TOKEN
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.load_lora_weights("artificialguybr/LogoRedmond-LogoLoraForSDXL-V2")  # Pode trocar para outro LoRA se quiser
    
    if verbose:
        print(f"✅ Modelo carregado em {time.time() - start:.2f}s\n")
    return pipe

def gerar_imagens_em_lote(prompts, quantidade=1, referencias=None, modelo=None, verbose=True, seed=None):
    import gc
    from tqdm import tqdm

    if not prompts or not isinstance(prompts, list):
        raise ValueError("O parâmetro 'prompts' deve ser uma lista não vazia de strings.")

    if modelo is None:
        if verbose:
            print("⚠️ Nenhum modelo fornecido. Carregando agora...")
        modelo = carrega_modelo(verbose=verbose)

    pastas_geradas = []

    for idx, prompt in enumerate(tqdm(prompts, desc="🚀 Processando Prompts")):
        if verbose:
            print(f"\n🎯 Prompt {idx + 1}/{len(prompts)}")

        nome_prompt = obter_nome_prompt_unico()
        pasta_saida = DIR_RESULTADOS / nome_prompt
        pasta_saida.mkdir(parents=True, exist_ok=False)

        total_start = time.time()
        json_path = pasta_saida / f"{nome_prompt}_info.json"

        # Criar estrutura básica do JSON ANTES da geração
        info = {
            "prompt": prompt,
            "modelo": "stabilityai/stable-diffusion-xl-base-1.0",
            "referencias": referencias if referencias else [],
            "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_imagens": quantidade,
            "imagens": []
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        for i in range(1, quantidade + 1):
            if verbose:
                print(f"🎨 Gerando imagem {i}/{quantidade}...")

            start = time.time()
            seed_atual = seed if seed is not None else torch.seed()
            generator = torch.manual_seed(seed_atual)

            imagem_info = {
                "index": i,
                "seed": seed_atual
            }

            try:
                imagem = modelo(prompt, generator=generator).images[0]
                nome_imagem = f"{nome_prompt}_img_{i}.png"
                caminho_imagem = pasta_saida / nome_imagem
                imagem.save(caminho_imagem)
                imagem_info["arquivo"] = str(caminho_imagem)
                if verbose:
                    print(f"   ✔ Imagem {i} salva em {time.time() - start:.2f}s")
            except Exception as e:
                imagem_info["erro"] = str(e)
                if verbose:
                    print(f"   ❌ Erro ao gerar imagem {i}: {e}")

            # Atualizar JSON com a imagem (ou erro)
            info["imagens"].append(imagem_info)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"📁 {quantidade} tentativas salvas em '{pasta_saida}'")
            print(f"⏱️ Tempo total para o prompt: {time.time() - total_start:.2f}s")

        pastas_geradas.append(str(pasta_saida))

        torch.cuda.empty_cache()
        gc.collect()

    return pastas_geradas

# ========== CONFIGURAÇÕES ==========
verbose = True
sleep_time = 0

if not verbose:
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    transformers_logging.set_verbosity_error()
    diffusers_logging.set_verbosity_error()
    diffusers_logging.disable_progress_bar()

# ========== PROMPTS ==========
prompts = [
    "vintage badge logo, minimalistic logo for a spiritual self-knowledge brand @tarot.bymile. Modern serif typography, clean vector layout, rosy magenta theme.",
    "Apothecary vintage logo style combining mystical and natural elements for @tarot.bymile — avoid literal icons, emphasize elegance and intuitive icons."
    "Combination mark logo for @tarot.bymile — blend text and abstract icon inspired by crescent moon or lantern. Warm and magical tone.",
    "Typography-based logo using soft handwritten font for @tarot.bymile. Feminine, vintage, inviting design with a mystical hint.",
    "vintage scalable logo with old school traces, suited for digital use. Color: #993d74. Include abstract symbol reflecting self-discovery for @tarot.bymile."
]

referencias_tratadas = tratar_imagens_referencia(verbose=True)
modelo = carrega_modelo(verbose=True)

pastas = gerar_imagens_em_lote(
    prompts=prompts,
    quantidade=2,
    referencias=referencias_tratadas,
    modelo=modelo,
    verbose=True,
    # seed=1234  # Seed opcional para reproduzir resultados
)

print("Pastas geradas:")
for p in pastas:
    print(f"✅ {p}")

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import pandas as pd

# def format_prompt(text): 
#     return f"Translate from English to Spanish:\nEnglish: {text}\nSpanish:"

# @torch.no_grad()
# def translate(model, tokenizer, texts, batch_size = 16, device = "cpu"):
#     translations = []
#     model.eval()
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]
#         prompts = [format_prompt(text) for text in batch_texts]
#         inputs = tokenizer(prompts, return_tensors = "pt", padding = True, truncation = True).to(device)

#         outputs = model.generate(
#             **inputs, 
#             max_new_tokens = max(map(lambda x: len(x), batch_texts)) + 50, 
#             temperature = 0.7, 
#             top_p = 0.9, 
#             do_sample = True, 
#             eos_token_id = tokenizer.eos_token_id
#         )

#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)
#         for full_output in decoded: 
#             translated_part = full_output.split("Spanish:")[-1].strip()
#             translations.append(translated_part)
#     return translations

# if __name__ == "__main__": 
#     df = pd.read_csv("annotation_in.csv")
#     english_inputs = df['corrected_english'].dropna().tolist()

#     # model_id = "meta-llama/Llama-3.2-1B"
#     model_id = "meta-llama/Llama-3.2-1B-Instruct"
#     model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto")
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id

#     device = (
#         "cuda" if torch.cuda.is_available() else 
#         "mps" if torch.backends.mps.is_available() else 
#         "cpu"
#     )

#     spanish_translations = translate(model, tokenizer, english_inputs, device = device)
#     df['translated_spanish'] = spanish_translations
#     df.to_csv("annotation_out.csv", index = False)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def format_prompt(text, lang="Spanish"):
    return f"[INST] Translate the following sentence from English to {lang}:\n{text} [/INST]"

@torch.no_grad()
def translate(model, tokenizer, texts, lang="Spanish", batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    translations = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        prompts = [format_prompt(text, lang=lang) for text in batch_texts]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.9,
            do_sample=False,  # Deterministic output is often better for translation
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the translated response after the instruction (can be improved with regex if needed)
        for full_output in decoded:
            if "[/INST]" in full_output:
                translated_part = full_output.split("[/INST]")[-1].strip()
            else:
                translated_part = full_output.strip().replace('\n', '')
            translations.append(translated_part)

    return translations

if __name__ == "__main__":
    df = pd.read_csv("annotation_in.csv")
    english_inputs = df['corrected_english'].dropna().tolist()

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    spanish_translations = translate(model, tokenizer, english_inputs, lang="Spanish")

    df['spanish_translation'] = spanish_translations
    df.to_csv("translated_output.csv", index=False)

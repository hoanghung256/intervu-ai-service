# Gemini models
GEMINI_GEMMA_3_27B_IT = "gemma-3-27b-it"
GEMINI_GEMMA_3_12B_IT = "gemma-3-12b-it"
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"

# Hugging Face router models
HUGGINGFACE_LLAMA_3_1_8B_NOVITA = "meta-llama/Llama-3.1-8B-Instruct:novita"
HUGGINGFACE_QWEN2_5_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
HUGGINGFACE_MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.3"
HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ = "meta-llama/Llama-3.3-70B-Instruct:groq"

# Backward-compatible aliases
GEMINI_DEFAULT_MODEL = GEMINI_GEMMA_3_27B_IT
HUGGINGFACE_DEFAULT_MODEL = HUGGINGFACE_LLAMA_3_1_8B_NOVITA

# Optional lists for validation/UI selection
GEMINI_SUPPORTED_MODELS = (
    GEMINI_GEMMA_3_27B_IT,
    GEMINI_GEMMA_3_12B_IT,
    GEMINI_2_0_FLASH,
    GEMINI_2_0_FLASH_LITE,
)

HUGGINGFACE_SUPPORTED_MODELS = (
    HUGGINGFACE_LLAMA_3_1_8B_NOVITA,
    HUGGINGFACE_QWEN2_5_7B_INSTRUCT,
    HUGGINGFACE_MISTRAL_7B_INSTRUCT,
    HUGGINGFACE_LLAMA_3_3_70B_INSTRUCT_GROQ,
)

GEMINI_MODEL_SET = set(GEMINI_SUPPORTED_MODELS)
HUGGINGFACE_MODEL_SET = set(HUGGINGFACE_SUPPORTED_MODELS)

PROVIDER_GEMINI = "gemini"
PROVIDER_HUGGINGFACE = "huggingface"


def resolve_provider_name(model):
    if not model:
        return None
    if model in GEMINI_MODEL_SET:
        return PROVIDER_GEMINI
    if model in HUGGINGFACE_MODEL_SET:
        return PROVIDER_HUGGINGFACE
    # HuggingFace model ids are namespaced ("org/name"); Gemini/Gemma ids are not.
    return PROVIDER_HUGGINGFACE if "/" in model else PROVIDER_GEMINI

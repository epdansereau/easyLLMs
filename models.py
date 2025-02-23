import yaml
from langchain_community.chat_models import ChatLiteLLM
from langchain_google_genai import ChatGoogleGenerativeAI


# Load settings from YAML file
def load_settings(preset):
    with open("chatbotsettings.yaml", "r") as file:
        presets = yaml.safe_load(file)
    if preset not in presets:
        raise ValueError(f"Preset '{preset}' not found in settings.yaml. Available presets: {list(presets.keys())}")
    return presets[preset]

# Load API keys from apikeys.yaml
def load_api_key(provider):
    with open("apikeys.yaml", "r") as file:
        api_keys = yaml.safe_load(file)
    if provider not in api_keys:
        raise ValueError(f"API key for provider '{provider}' not found in apikeys.yaml.")
    return api_keys[provider]

# Get the model instance
def get_model(settings):
    api_key = load_api_key(settings["custom_llm_provider"])
    if settings["custom_llm_provider"] == "gemini":
        '''current implementation will return tool results but not tool calls.'''
        return ChatGoogleGenerativeAI(
            model=settings["model"], 
            api_key=api_key
        )
    return ChatLiteLLM(
        model=settings["model"], 
        custom_llm_provider=settings["custom_llm_provider"], 
        api_base=settings.get("api_base"), 
        api_key=api_key
    )
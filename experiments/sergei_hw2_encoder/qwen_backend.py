"""Lazy PyTorch/Transformers backend for the frozen Qwen selector."""
from dataclasses import dataclass
from time import perf_counter

from .encoder import RecoverableInferenceError


@dataclass(frozen=True)
class Generation:
    raw: str
    input_tokens: int
    generated_tokens: int
    generation_seconds: float


class GenerationError(RecoverableInferenceError):
    def __init__(self, message, input_tokens=None, generation_seconds=None):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.generation_seconds = generation_seconds


def load(config, device, dtype, allow_download):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('PyTorch MPS is unavailable')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('PyTorch CUDA is unavailable')

    def synchronize():
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    torch.manual_seed(config['seed'])
    kwargs = {'revision': config['revision'], 'local_files_only': not allow_download,
              'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(config['model'], **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        config['model'], torch_dtype=getattr(torch, dtype), use_safetensors=True, **kwargs)
    model = model.eval().to(device)
    synchronize()

    def generate(prompt):
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=config['enable_thinking'])
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(device)
        input_tokens = inputs['input_ids'].shape[-1]
        started = perf_counter()
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs, do_sample=config['do_sample'], num_beams=config['num_beams'],
                    max_new_tokens=config['max_new_tokens'], num_return_sequences=1,
                    temperature=None, top_p=None, top_k=None)
            synchronize()
        except RuntimeError as error:
            # Backend/device failures terminate the run. A per-call timeout can use fallback.
            raise RuntimeError(f'Qwen generation stopped: {error}') from error
        except TimeoutError as error:
            raise GenerationError(str(error), input_tokens, perf_counter() - started) from error
        elapsed = perf_counter() - started
        tokens = output[0, input_tokens:].tolist()
        # Remove only the generation terminator. Preserve any other emitted markers.
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        content = tokens[:-1] if tokens and tokens[-1] in eos else tokens
        raw = tokenizer.decode(content, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
        return Generation(raw, input_tokens, len(tokens), elapsed)

    return generate

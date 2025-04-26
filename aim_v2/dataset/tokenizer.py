from transformers import AutoTokenizer


def load_hf_tokenizer(checkpoint='path/to/hf_model'):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = 'right'

    return tokenizer


if __name__ == '__main__':
    import os

    cur_path = os.path.dirname(os.path.abspath(__file__))
    llama3_tokenizer_config_path = os.path.join(cur_path, 'llama3_tokenizer_config')
    tokenizer = load_hf_tokenizer(checkpoint=llama3_tokenizer_config_path)
    print(' - Vocab size: ', tokenizer.vocab_size)

    # 英语实例
    text = "Hey, Ross, let me tell you something."
    print(' - Example-1: ', text)
    token_ids = tokenizer.encode(text, )
    print(' - String to token ID: ', token_ids)

    text = tokenizer.decode(token_ids)
    print(' - Token ID to string: ', text)

    print()
    
    # 中文实例
    text = "好吧，罗斯，让我好好给你上一课。"
    print(' - Example-2: ', text)
    token_ids = tokenizer.encode(text, )
    print(' - String to token ID: ', token_ids)

    text = tokenizer.decode(token_ids)
    print(' - Token ID to string: ', text)

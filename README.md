# LLM-Watchdog
Part of the dissertation "Assessing robustness of large language models to data extraction attacks".
Tool for blackbox testing of LLMs (MVP).

## Restrictions on Content Use for Training

- **Copyrighted Documents**: Any text protected by copyright without the owner's consent. This may include literary works, scientific articles, software, and even photographs that cannot be used without the appropriate license.

- **Confidential Information**: Documents containing personal data or other confidential information, such as medical records, financial data, or safety-related documents, should not be used for training.

- **Inappropriate content**: Materials that promote violence, hatred, discrimination, or other forms of negative impact on society should not be part of the training dataset. This includes extremist texts, racist remarks, and other forms of hate.

- **Incorrect or misleading information**: Texts that contain blatant falsehoods, pseudoscientific claims, or conspiracy theories can lead to problems in how the model generates information and interacts with users.

- **Secret or Internal Documents of Organizations**: Texts belonging to companies or governmental institutions that are classified as internal or secret should not be included in the training set.

- **User-Generated Content from Social Media**: In the absence of clear policies regarding the use of user-generated content, such data should not be used, as it may raise intellectual property rights issues.

You can control this parameter via the *--content-type* key. Note that you can evaluate either a single type or all types at once.

## Set of Scenarios

- **Paper_0**: https://llm-attacks.org/index.html#examples
- **Paper_1**: Data Extraction Techniques: Augment The LLM API With LLM Gateway and Regex
- **Paper_2**: https://github.com/mik0w/pallms

## Instalation

## API keys

If you want to use OpenAI or YandexGPT or GigaChat, you need to set your API keys.

```bash
# For OpenAI models
export OPENAI_API_KEY="your-openai-key"

# For YandexGPT models
export YANDEXGPT_API_KEY="your-key"

# For GigaChat models
export GIGACHAT_API_KEY="your-key"
```

## Usage and Feedback

```bash
# Get list of available models:
python3 llmwatchdog.py --avail

# For ollama models
python3 llmwatchdog.py --model llama3.1 --model-type ollama

# For OpenAI models
python3 llmwatchdog.py --model gpt-4o-mini --model-type openai --output myfile2

# For Sber models 
# Note: https://developers.sber.ru/docs/ru/gigachat/certificates if you got error like [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain
python3 llmwatchdog.py --model gigachat --model-type sber --content-type content-type/mis-info --scenarios suffix.txt

# for Yandex models
python3 llmwatchdog.py --model yandexgpt --model-type yandex --output myfile.txt

```


## License

## Thanks

API for getting list of available ollama models https://github.com/webfarmer/ollama-get-models
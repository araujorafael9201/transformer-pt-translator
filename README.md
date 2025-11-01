# Transformer-based Machine Translation (English to Portuguese)

This project is an implementation of the Transformer model for machine translation, as described in the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. The goal of this project is to translate text from English to Portuguese.

## Project Structure

- **`data/`**: Contains the training data (`en.txt` and `pt.txt`).
- **`utils/dataloader.py`**: Contains the data loading and preprocessing logic.
- **`model.py`**: Defines the Transformer model architecture.
- **`train.py`**: Script to train the Transformer model.
- **`inference.py`**: Script to run inference on a trained model.
- **`requirements.txt`**: A file that contains all the dependencies for the project.

## How to Use

### 1. Setup the Environment

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Train the Model

To train the model, run the `train.py` script:

```bash
python train.py
```

You can also customize the training process by providing command-line arguments. For example:

```bash
python train.py --epochs 50 --batch_size 128
```

The full list of arguments can be found in [Training Parameters](#training-parameters)

### 4. Run Inference

To translate a sentence, use the `inference.py` script. Note that `translator_model.pth` is an example path to a trained model; you will need to train your own model or provide a path to an existing one. You can download a trained version of this implementation on Hugging face, as mentioned in the [Translation Examples](#translation-examples) section.

```bash
python inference.py "your sentence here" --model_path path/to/your/model.pth
```

## Training Parameters

The following table details the command-line arguments available for customizing the training process in `train.py`:

| Argument             | Description                                                     | Default Value                        |
| -------------------- | --------------------------------------------------------------- | ------------------------------------ |
| `embed_size`         | The embedding dimension size.                                   | `128`                                |
| `compile_model`      | If set, pre-compiles the model with `torch.compile`.            | `False`                              |
| `epochs`             | The number of training epochs.                                  | `100`                                |
| `max_seq_len`        | The maximum length of input data.                               | `128`                                |
| `max_dataset_size`   | Optionally limits the dataset size for faster training.         | `10000000`                           |
| `batch_size`         | The effective batch size for training after gradient accumulation. | `256`                                |
| `gpu_batch_size`     | The batch size supported by the GPU.                            | `64`                                 |
| `learning_rate`      | The learning rate for the optimizer.                            | `3e-4`                               |
| `en_file`            | The path to the English source file.                            | `data/en.txt`                        |
| `pt_file`            | The path to the Portuguese target file.                         | `data/pt.txt`                        |
| `model_path`         | The path to save the trained model.                             | `translator_model.pth`               |
| `checkpoint_path`    | The path to save checkpoints during training.                   | `translator_model_checkpoint.pth`    |

## Translation Examples

Here are some examples of translations from a previously trained model, available on [Hugging Face](https://huggingface.co/rafaelaraujo13/transformer-translator-english-portuguese):

| English                         | Portuguese                 |
| ------------------------------- | -------------------------- |
| Hello, how are you?             | Como é que vocês são?      |
| What is your name?              | O que é o seu nome?        |
| I am a software engineer.       | Sou engenheiro de software.|
| This is a test.                 | Este é um teste.           |
| I love to code.                 | Eu adoro o código.         |
| The weather is beautiful today. | O tempo é lindo hoje.      |

This model was trained with the following parameters:

| Parameter          | Value      |
| ------------------ | ---------- |
| `embed_size`       | 128        |
| `compile_model`    | True      |
| `epochs`           | 50         |
| `max_seq_len`      | 32         |
| `max_dataset_size` | 150000     |
| `batch_size`       | 256        |
| `gpu_batch_size`   | 64         |
| `learning_rate`    | 3e-4       |

Training took approximately 2 hours on an NVIDIA A40 GPU with 48GB RAM. Note that `max_seq_len` is 32, so to run it:
```bash
python inference.py "your sentence" --max_seq_len 32
```
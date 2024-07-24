# all parameters:
import os
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoFeatureExtractor, SwinForImageClassification
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from funcs import *


class PARAMS:
    seed = 3407  # Torch.manual_seed(3407) is all you need

    model_path = "microsoft/swin-base-patch4-window12-384"
    model_output_dir = "fine_tuned_models/vit_finetuned"

    train_batch_size = 16
    val_batch_size = 64
    logging_dir = 'logs/swin'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    dataset = load_dataset("imagefolder", data_dir="data/datasetfolder/")
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    # image_processor = AutoImageProcessor.from_pretrained(PARAMS.model_path)
    image_processor = AutoFeatureExtractor.from_pretrained(PARAMS.model_path)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    train_transforms = Compose(
        [
            Resize(size),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch


    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    dataset['train'].set_transform(preprocess_train)
    dataset['validation'].set_transform(preprocess_val)
    dataset['sampled_val'] = dataset['validation'].shuffle(seed=42).select(range(10000))
    dataset['sampled_val'].set_transform(preprocess_val)

    model = SwinForImageClassification.from_pretrained(
        PARAMS.model_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        PARAMS.model_output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",  # 设置为"steps"
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=PARAMS.train_batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=PARAMS.val_batch_size,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_dir=PARAMS.logging_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        eval_steps=100,  # 设置评估频率为每100个step
        push_to_hub=False
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['sampled_val'],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

import argparse
from datasets import load_dataset
from transformers import TrainingArguments
from cnn.prelu_cnn import CNN, CNNTrainer, preprocess_images
from training.trainer import ModelTrainer


def main(use_prelu, use_builtin_conv, num_epochs):
    """Main function to orchestrate the training and evaluation of the CNN."""

    print("Loading CIFAR-10 dataset...")
    dataset = load_dataset("cifar10")

    print(f"Creating CNN model (PReLU: {use_prelu}, Builtin conv: {use_builtin_conv})...")
    model = CNN(use_prelu=use_prelu, use_builtin_conv=use_builtin_conv)

    training_args = TrainingArguments(
        output_dir=f"./results/cnn_results_{'prelu' if use_prelu else 'relu'}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs/logs_{'prelu' if use_prelu else 'relu'}",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        evaluation_strategy="epoch",
    )

    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        preprocess_fn=preprocess_images,
        trainer_class=CNNTrainer,
    )

    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument(
        "--use_prelu",
        action="store_true",
        default=False,
        help="Use PReLU activation instead of ReLU.",
    )
    parser.add_argument(
        "--use_builtin_conv",
        action="store_true",
        default=True,
        help="Use built-in torch.nn.functional.conv2d.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs."
    )
    args = parser.parse_args()

    main(
        use_prelu=args.use_prelu,
        use_builtin_conv=args.use_builtin_conv,
        num_epochs=args.num_epochs,
    ) 
from transformers import Trainer, TrainingArguments
from datetime import datetime


class ModelTrainer:
    """
    A generic class to handle the boilerplate of training a model using Hugging Face Trainer.
    It takes all the necessary components for training and orchestrates the process.
    """

    def __init__(
        self,
        model,
        training_args: TrainingArguments,
        train_dataset,
        eval_dataset,
        preprocess_fn=None,
        data_collator=None,
        trainer_class=Trainer,
        compute_metrics=None,
        compute_loss=None,
        callbacks=None,
        resume_from_checkpoint=None,
        disable_timestamped_logging=False,
    ):
        """
        Initializes the ModelTrainer.

        Args:
            model (torch.nn.Module): The model to train.
            training_args (TrainingArguments): Configuration for the training process.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            preprocess_fn (callable, optional): A function to preprocess the datasets. Defaults to None.
            data_collator (callable, optional): The data collator. Defaults to None.
            trainer_class (Trainer, optional): A custom Trainer class to use. Defaults to `transformers.Trainer`.
            compute_metrics (callable, optional): Function to compute metrics during evaluation. Defaults to None.
            compute_loss (callable, optional): Custom loss function. Defaults to None.
            callbacks (list, optional): List of callbacks for the trainer. Defaults to None.
            resume_from_checkpoint (str, optional): Path to checkpoint to resume from. Defaults to None.
            disable_timestamped_logging (bool, optional): If True, disables automatic timestamped logging directories. Defaults to False.
        """
        self.model = model
        self.training_args = training_args
        self.train_dataset_raw = train_dataset
        self.eval_dataset_raw = eval_dataset
        self.preprocess_fn = preprocess_fn
        self.data_collator = data_collator
        self.trainer_class = trainer_class
        self.compute_metrics = compute_metrics
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.resume_from_checkpoint = resume_from_checkpoint
        self.disable_timestamped_logging = disable_timestamped_logging

        # Add timestamp to logging_dir to prevent overwriting logs (unless disabled)
        if self.training_args.logging_dir and not disable_timestamped_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.training_args.logging_dir = (
                f"{self.training_args.logging_dir}_{timestamp}"
            )

    def run(self):
        """
        Executes the training and evaluation process.
        """

        # Only initialize the trainer once and store it
        if not hasattr(self, "trainer"):
            self.trainer = self.trainer_class(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset_raw,
                eval_dataset=self.eval_dataset_raw,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=self.callbacks,
            )

        print("Starting training...")
        train_results = self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        print("Evaluating model...")
        eval_results = self.trainer.evaluate()

        print(f"Train results: {train_results}")
        print(f"Eval results: {eval_results}")
        return self.trainer, train_results, eval_results 
    
    def evaluate(self):
        """
        Evaluates the model using the existing trainer instance.
        """
        if not hasattr(self, "trainer"):
            self.trainer = self.trainer_class(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset_raw,
                eval_dataset=self.eval_dataset_raw,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=self.callbacks,
            )

        print("Evaluating model...")
        eval_results = self.trainer.evaluate()
        return eval_results
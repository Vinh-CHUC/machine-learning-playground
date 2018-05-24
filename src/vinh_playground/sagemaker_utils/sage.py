import argparse
import os

import boto3
import sagemaker

def build_inputs(base_path):
    return {k: os.path.join(base_path, k) for k in ["train", "val", "test", "misc"]}


def train(job_name):
    base_path = (
        "s3://lyst-data-science-sagemaker/datasets/dali/"
        "pairs_agreement-0.5_ratio-80-10-10_pos-neg-ratio-1.0_date-20180517"
    )

    estimator = sagemaker.estimator.Estimator(
        image_name=os.getenv("ECR_REPO") + ":latest",
        output_path="s3://lyst-data-science-sagemaker/vinh",
        role="SageMakerFull",
        train_instance_count=1,
        train_instance_type="ml.p2.xlarge",
        train_volume_size=30,
        hyperparameters={
           "inputs_architecture": 'siamese',
           "loss_type": 'cross_entropy',
           "val_period": 3000,
           "use_short_text": 1,
           "use_images":1,
           "use_price":1,
        }
    )
    estimator.fit(build_inputs(base_path), job_name=job_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SageMaker training job")
    args = parser.parse_args("-n", "--job_name", help="The SageMaker training job name")
    args = parser.parse_args("-i", "--image_name", help="The docker image name")

    train(args.job_name, args.image_name)

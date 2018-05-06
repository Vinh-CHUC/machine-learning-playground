import argparse

import boto3
import sagemaker

S = boto3.Session()

def build_image_uri(image_name):
    account_id = S.client('sts').get_caller_identity()['Account']
    region = S.region_name

    return f'{account_id}.dkr.ecr.{region}.amazonaws.com/{image_name}:latest'


def train(job_name, image_name):
    estimator = sagemaker.estimator.Estimator(
        image_name=build_image_uri(image_name),
        role="vinh_sagemaker",
        train_instance_count=1,
        train_instance_type="ml.p2.xlarge",
        train_volume_size=30,
        hyperparameters={"epochs": 10}
    )
    estimator.fit("s3://sagemaker-eu-west-1-187232669044/data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SageMaker training job")
    args = parser.parse_args("-n", "--job_name", help="The SageMaker training job name")
    args = parser.parse_args("-i", "--image_name", help="The docker image name")

    train(args.job_name, args.image_name)

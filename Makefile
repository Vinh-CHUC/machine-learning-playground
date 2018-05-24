build:
	docker build -t $${ECR_REPO}:latest .

push:
	docker push $${ECR_REPO}:latest

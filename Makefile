build:
	docker build -t vinh:latest .

push:
	docker tag vinh:latest 187232669044.dkr.ecr.eu-west-1.amazonaws.com/vinh:latest
	docker push 187232669044.dkr.ecr.eu-west-1.amazonaws.com/vinh:latest

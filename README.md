# mle-framework
Following tutorial to build Machine Learning Framework while adapting it to my needs.

The original tutorial can be found here: https://towardsdatascience.com/a-framework-for-building-a-production-ready-feature-engineering-pipeline-f0b29609b20f


To execute the ETL and the training:

1. Clone the repo
2. Build the Docker image
```
docker built -t <your_image_name> .
```
3. Run the following command
```
docker run <your_image_name> python main.py
```

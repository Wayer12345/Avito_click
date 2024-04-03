Classification task, which target is whether or not client will click on contextual ad. All data can be received from kaggle - https://www.kaggle.com/c/avito-context-ad-clicks. 
As a result after data merging and preprocessing was trained Catboost model, which was put in docker container and deployed using FastAPI.

# How to run
- Install and run Docker
- Build Docker image using: docker build . -t avito_click_inference -f dockerfile.txt
- Run Docker container using: docker run --rm -p 80:80 avito_click_inference
- Go to http://localhost:80 to see, if connection is clear
- Go to http://localhost:80/docs to see all available methods of the API
- Run api_request.py to get prediction for an object (or some diffrent objects)

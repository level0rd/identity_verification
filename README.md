# Identity confirmation by photo with Dlib models.

## Build and run the application locally

```bash
streamlit run person_identity.py
```

## Build and run the application in Docker
```bash
# Build Docker image
docker build . -t streamlit
# Run Docker container
docker run -p 8501:8501 streamlit
```
If the Docker container is running locally, the application will be available at http://localhost:8501.

## Architecture
<p align="center">
  <img width="4096" alt="Arch1_7" src="https://github.com/level0rd/identity_verification/assets/45522296/76275151-28e2-4633-9a58-36b27edb31a0.png">
</p>

## Example
<p align="center">
  <img width="1000" alt="Arch2" src="https://github.com/level0rd/identity_verification/assets/45522296/3a9e4100-4511-462b-8309-dba736432db2.jpg">
  <img width="1000" alt="Sam" src="https://github.com/level0rd/identity_verification/assets/45522296/0717d27d-8538-47ec-a161-eed39e8350c4.png">
</p>

### Successful identity confirmation
https://github.com/level0rd/identity_verification/assets/45522296/a422b954-6b78-4b0a-be6f-22b785123c7e


### Failed identity confirmation
https://github.com/level0rd/identity_verification/assets/45522296/22494672-159d-4495-beb1-d0237739c2ba



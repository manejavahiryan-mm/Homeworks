## Project Overview

This project analyzes the potential diffusion of Oura Ring 4 using the Bass Diffusion Model. Fitbit's historical registered-user data (2014–2025) are used to estimate the model parameters (p), (q), and (M). The estimated parameters are then applied to forecast Oura Ring 4 adoption from 2025 to 2030, using Oura's reported sales as a calibration reference.

## Project Files

bass_model.ipynb — Python analysis, calculations, predictions, and explanations.

data/ — Historical Fitbit dataset downloaded from Statista.

report/ — Final PDF report.

img/ — Directory for project images, if applicable.

## How to Run

Open bass_model.ipynb in VS Code with Python and Jupyter installed, and select Run All.

Required Python libraries: numpy, pandas, and scipy.

## References

Alderton, M. (2025, October 9). *Oura Ring 4*. TIME. https://time.com/collections/best-inventions-2025/7318554/oura-ring-4/

Fitbit, & Business of Apps. (2026, March 9). *Number of registered users of Fitbit from 2014 to 2025 (in millions)* [Graph]. Statista. https://www.statista.com/statistics/1327371/fitbit-registered-users/

Oura Inc. (2026, September 3). *Registration statement on Form S-1*. U.S. Securities and Exchange Commission. https://www.sec.gov/Archives/edgar/data/2133022/000119312526381855/d119865ds1.htm
# DL-based-Cough-Sneeze-Detector
Keras implementation of NLP based cough and sneeze detector based on GRU.

This model has been designed to detect coughing and sneezing sounds by listening to ambient audio on an embedded/mobile hardware using any microphone connected to it. This can run in the background as an app on mobile users given appropriate permissions or as listening pods in public areas where large number of people congregate. Considering the COVID-19 pandemic and future scenarios, this can be used to detect any anomalies in the cough/sneeze baseline count of the general public and raise location and time specific alerts. The system can be designed to maintain user anonymity while capturing data and can run a completely on-device processing without cloud/network audio upload. The application will communicate the statistics to the central server following a pre-configured interval.

<p align="center">
  <img title="gather data" alt="location alerts" width="60%" height="60%" src="https://github.com/cobaltKite/DL-based-Cough-Sneeze-Detector/blob/master/imgs/displayImage1.jpg?raw=true">
  <p align="center"><strong>Unobtrusive data collection</strong></p>  
</p>

<br>
<figure>
  <img title="analytics" alt="location alerts" width="100%" height="100%" src="https://github.com/cobaltKite/DL-based-Cough-Sneeze-Detector/blob/master/imgs/map3.jpg?raw=true">
  <p align="center"><strong>Cough/sneeze detection heatmaps</strong></p>  
</figure>


The model is implemented using Keras with Tensorflow backend. The implementation uses the Gated Recurrent Unit to detect cough/sneeze sounds. 

<br>

<p align="center">
  <img title="Gated Recurrent Unit Architecture - simplified diagram" alt="gated recurrent unit" width="65%" height="65%" src="https://upload.wikimedia.org/wikipedia/commons/3/37/Gated_Recurrent_Unit%2C_base_type.svg?raw=true">
<p align="center"><strong>GRU cell</strong></p>
</p>

The model is trained on Sick Sounds Dataset (https://osf.io/tmkud/). The model can be trained with more data to reject background noise better, and have higher sensitivity and specificity to cough and sneeze sounds occuring under different background noise conditions.

## Proposed Solution

<p align="center">
  <img title="Functional Architecture" alt="proposed solution" width="100%" height="100%" src="https://github.com/cobaltKite/DL-based-Cough-Sneeze-Detector/blob/master/imgs/flowChart_Idea.png?raw=true">
<p align="center"><strong>Functional Architecture</strong></p>
 
</p>
The proposed solution consists of:

### i. Data Injection Platform
Cough/sneeze data can be collected either from voluntary installation of mobile apps by individual users or from public installations of microphone enabled IoT devices at public places. Mobile app users can anonymously share cough/sneeze detection information along with approximate location when sufficient permissions are granted on the device. All audio processing will be done on the edge(device) and no audio clips or streams are sent from the device to the server. Anonymous actionable information is sent to the central server at configurable intervals. The battery operated IoT devices installed at known public locations of high footfall with a GPRS connection which can transmit low bandwidth metadata streams to the server.

### ii. Central Server (archival and insight generation)
The central server collects and archives the registered cough and sneeze counts, times and locations over a large enough duration to create a baseline cough/sneeze pattern. The raw data can be used as-is or can be used along with the historical predictions learnt on-the-go to detect anomalies. 

### iii. Analytics visualization
The data can be used in conjuction with population density, demographical information etc. for use by healthcare professionals to respond faster and better to changing pandemic trends, and for early detection of new disease symptoms.

## Pre-requisites:
Keras==2.3.1
matplotlib==3.2.1
numpy==1.18.4
pydub==0.23.1
scikit-learn==0.22.2
tensorflow==2.2.0

For the complete list of requirements install requirements using pip install -r requirements.txt

## Instructions to run
Run python3 cough_sneeze_inference.py to run the sample input. Detection output is displayed in the terminal.



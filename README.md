# DL-based-Cough-Sneeze-Detector
Keras implementation of NLP based cough and sneeze detector based on GRU.

This model has been designed to detect coughing and sneezing sounds by listening to ambient audio on an embedded/mobile hardware using any microphone connected to it. This can run in the background as an app on mobile users given appropriate permissions or as listening pods in public areas where large number of people congregate. Considering the COVID-19 pandemic and future scenarios, this can be used to detect any anomalies in the cough/sneeze baseline count of the general public and raise location and time specific alerts. The system can be designed to maintain user anonymity while capturing data and can run a completely on-device processing without cloud/network audio upload. The application will communicate the statistics to the central server following a pre-configured interval.

The model is implemented using Keras with Tensorflow backend. The implementation uses the Gated Recurrent Unit to detect Trigger words. 

The model is trained on Sick Sounds Dataset (https://osf.io/tmkud/). The model can be trained with more data to reject background noise better, and have higher sensitivity and specificity to cough and sneeze sounds occuring under different background noise conditions.

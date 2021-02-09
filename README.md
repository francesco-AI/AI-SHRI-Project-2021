# AI-SHRI-Project-2021
The SHRI project for AI exam with Prof. Nardi

# Artificial Intelligence [12 Credit] 
### Master in Artificial Intelligence and Robotics
### - Prof. Nardi -

#### SHRI Project
#### Student: Francesco Cassini
#### Sapienza ID number: 785771
#### GITHUB directory:  https://github.com/francesco-AI/AI-SHRI-Project-2021


## Abstract

For my Spoken Human-Robot Interaction project I have implemented a bot that is able of performing a series of actions through vocal commands via Telegram Bot API:  one of the most interesting feature of my work is the control of a remote system. In fact I have installed it on a Nvidia Jetson Nano connected to internet to receive my commands in any moment of day.

From what concern the study of Dialogue manager, I focused my attention on the "intent recognition" block. In fact, I have implemented a sequential solution in which the "request" of the user is first analyzed by a "simple" circuit that uses the grammatical construct looking for matches in the dictionary of actions.
If it fails, it passes the request to a neural network which is trailed on the examples provided today using the system. In this sense, the idea behind is to utilizes my system to collect specific data for the training of this second network.

I was inspired by my daily use of a “Save folder” Telegram function, which allows to save material of interest (links, photos, files) in a special “folder”: it works like a “primitive” NAS (Network Attached Storage) system, but it's completely lacking of a “smart”  search function.
The chosen context allowed me to implement a series of dialogues through which to experiment the use of speech-recognition and text-to-speech routines: through the "Google Speech Recognition" and "SAY" libraries it’s possible to send and receive voice commands (in English).


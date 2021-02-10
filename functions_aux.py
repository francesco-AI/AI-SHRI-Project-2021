##############################################################################
##############################################################################
########                    SHRI PROJECT                       ###############
########                    Prof. Nardi                        ###############
########                                                       ###############
########            Student: FRANCESCO CASSINI                 ###############
########            Sapienza ID: 785771                        ###############
########     Master in Roboics and Artificial Intelligence     ###############
##############################################################################
##############################################################################
########    
##############################################################################
##############################################################################

import json 
import socket
from socket import gethostbyname


def create_dictio(file_input):
    dictionary = {'prova':'ciao'}
    dictionary = json.dumps(dictionary)
    with open(file_input, 'w+') as f: 
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
        f.close()


def saving_function(file_archive, input_object, type_object):

    dictionary_archive = {}
    with open(file_archive, encoding='utf-8', errors='ignore') as json_data:
        dictionary_string = json.load(json_data)
        dictionary_archive = json.loads(dictionary_string)

    if type_object in dictionary_archive:
        for element in input_object:
            dictionary_archive[type_object].append(element)
    else:
        dictionary_archive[type_object] = input_object

    print(dictionary_archive)
    dictionary_archive = json.dumps(dictionary_archive)
    with open(file_archive, 'w+') as f: 
        json.dump(dictionary_archive, f, ensure_ascii=False, indent=4)
        f.close()



def reading_function(file_archive, type_object):

    dictionary_archive = {}
    with open(file_archive, encoding='utf-8', errors='ignore') as json_data:
        dictionary_string = json.load(json_data)
        dictionary_archive = json.loads(dictionary_string)

    if type_object in dictionary_archive:
        return dictionary_archive[type_object][-1]


def ipaddress_function():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip



def saving_injsonformat_phrase(file_archive, phrase, label):
    dictionary_archive = {}
    with open(file_archive, encoding='utf-8', errors='ignore') as json_data:
        dictionary_string = json.load(json_data)
        dictionary_archive = json.loads(dictionary_string)

    if not(phrase in dictionary_archive):
        dictionary_archive[phrase] = label

    print(dictionary_archive)
    dictionary_archive = json.dumps(dictionary_archive)
    with open(file_archive, 'w+') as f: 
        json.dump(dictionary_archive, f, ensure_ascii=False, indent=4)
        f.close()


def saving_phrase(file_archive, phrase, label):
    f=open(file_archive, "a+")

    text = '{"text": "' + str(phrase) + '", "language": "EN", "label": "' + str(label) + '"}' + '\n'
    #{"text": "John è andato a farso jgh", "language": "EN", "label": "save"}
    #dictionary_archive[phrase] = label
    f.write(text)
    f.close()

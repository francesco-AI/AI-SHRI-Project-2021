#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=W0613, C0116
# type: ignore[union-attr]
# This program is dedicated to the public domain under the CC0 license.

"""
First, a few callback functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is AGENT_REPLYed and runs until we press Ctrl-C on the command line.
Usage:
Example of a bot-user conversation using ConversationHandler.
Send /AGENT_REPLY to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""
import os
import speech_recognition as sr
from gtts import gTTS
import time
import calendar
import subprocess
from ctypes import *
from contextlib import contextmanager
import pyaudio
import playsound

# import stanfordnlp
import stanza

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)
import validators
import librosa
import soundfile as sf
import functions_aux as fun_aux

AGENT_REPLY, HEAR_USER, EXECUTE_INTENT, CONFIRM_KEYBOARD, QUESTION_KEYBOARD, RESPONSE_TO_PHOTO, RESPONSE_TO_VOICE, LOCATION, BIO = range(9)

MODELS_DIR = '.'
DOWNLOAD_DIR = 'download/'
ARCHIVE_DIR = 'archive.json'

# stanfordnlp.download('it', MODELS_DIR) # Download the Italian models
# nlp = stanfordnlp.Pipeline(models_dir=MODELS_DIR, treebank='it_isdt')

stanza.download('en') # Download the Italian models
nlp = stanza.Pipeline('en')

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


Agent = {'intent': None, 'prev_state': None, 'state': None, 'state_parameter': {'step':None, 'variable': None},'phrase': 'I\'m ready to ask your requests', 'time_last_speak': None, 'next_state':''} 
#Agent_state = {'finding_parameters':['finding_parameters', 'objects_receiving', 'checking_globalvar','asking_globalvar', 'executing']}

User = {'intent':'unknown', 'key':'', 'keyboard': None, 'keyboard_regex': None}
Global_objects = {'photo': [], 'file': [], 'link':[], 'document': [], 'videonote': [], 'video': [], 'audio':[]}
Objects_type = ['photo', 'file', 'link', 'document', 'video', 'videonote', 'audio']

Global_variables = {'objects_receiving': 0, 'last_received': None}
#Global_objects = {'photo':[], 'file':[], 'link':[], 'document':[], 'audio':[], 'video':[], 'text':[]}

tags = {'photo':0, 'file':0, 'link':0, 'document':0, 'audio':0, 'video':0, 'text':0}
#User['intent'].append('unknown')
History_variables = []
History_variables_index = {}
debug = True

#https://www.analyticsvidhya.com/blog/2019/02/stanfordnlp-nlp-library-python/
VB_dict = {
'VERB': 'verb, base form take',
'V': 'verb, base form take',
'VBD': 'verb, past tense took',
'VBG': 'verb, gerund/present participle taking',
'VBN': 'verb, past participle taken',
'VBP': 'verb, sing. present, non-3d take',
'VBZ': 'verb, 3rd person sing. present takes'
}


Action_dict = {
# verb  - intent_identifier, number of params to require, affermative phrase, negative pharse, interrogative phrase
'unknown': {'phrase_positive': 'I did not understood: can you repeat please?'},
'satisfied': {'phrase_positive': 'I have all the parameters that I need. I can do what have you ask me!'},
'query_exe': {'phrase_positive': "What I've to do?"},
'photo_received': {'phrase_positive': "Photo received"},
'save': {'params': {'objects': None,'tags': None}, 'exe': 'saving_document','phrase_positive': 'Let\'s start Saving function... Done! I\'m ready for a new job...', 'description': 'This function will save docu/photo/link and whatever else, giving them a tag. You can recover them, asking to system'},
'read': {'params': {'tags': None}, 'exe': 'reading_document','phrase_positive': 'I hope to have found what you want... anyway I\'m ready for a new job...', 'description': 'This function will save docu/photo/link and whatever else, giving them a tag. You can recover them, asking to system'},
'send': {'params': None, 'exe': 'sending_IPhost','phrase_positive': 'Send IP host machine address', 'description': 'This function will send you the host address of machine on which this telegram bot are running',},
'command ubuntu terminal': {'params': None, 'exe': 'sending_IPhost','phrase_positive': 'Send IP host machine address', 'description': 'This function will send you the host address of machine on which this telegram bot are running',},
'password_reseend': {}

}

Params_dict = {
# verb  - intent_identifier, number of params to require, affermative phrase, negative pharse, interrogative phrase
'objects': {'request': 'Which object(s) do you want I?', 'found_in_globalvariable': 'Previously you gave ', 'ask_confirmation': 'Previously you gave this in input. Are they?', 'received': 'I have received '},
'file': {'request': 'Which file/files do you want I?', 'action': 'request_input_file'},
'tags': {'request': 'Which tags I have to use?', 'action': 'request_input_tags', 'ask_confirmation': 'Please confirm before to proceed'},
'exe' : {'request': 'Which python function I have to associate to this action?', 'action': 'request_exe'},
'folder' : {'request': 'This exe requires some parameters to work? Please, indicate them', 'action': 'request_params'},
'new_exe' : {'request': 'Which new python function I have to associate to this action?', 'action': 'request_newexe'},
'params' : {'request': 'This exe requires some parameters to work? Please, indicate them', 'action': 'request_params'},
'new_params' : {'request': 'This exe requires some parameters to work? Please, input them, separated by virgula', 'action': 'request_newparams'},
}



Keyboard_dict =  {'boolean':[[['Yes', 'No']],'^(Yes|No)$'], 'check_alternatives':[[['Yes', 'No', 'Others']],'^(Yes|No|Others)$']  }

########################################################
#####    AGENT SPEAKS, REPLY, ASK, AND QUERY INPUT
########################################################
### TELEGRAM ACTIONS


def agent_action(update: Update, context: CallbackContext) -> int:
    global Agent, User, Global_objects, Global_variables
    if Agent['phrase'] != None:
        phrase = "\"" + str(Agent['phrase']) + "\""
        print(phrase)
        speak_simple(phrase)
        audio_file = open('Agent_reply.wav', 'rb')
        update.message.reply_audio(audio_file)
        update.message.reply_text(phrase,reply_markup=ReplyKeyboardRemove())
    Agent['phrase'] = Action_dict['query_exe']['phrase_positive']
    return HEAR_USER

def agent_show(item, update: Update, context: CallbackContext) -> int:
    update.message.reply_text(item,reply_markup=ReplyKeyboardRemove())



def agent_ask_keyboard(update: Update, context: CallbackContext) -> int:
    global Agent, User, Global_objects, Global_variables
    if Agent['phrase'] != None:
        phrase = "\"" + str(Agent['phrase']) + "\""
        print(phrase)
        speak_simple(phrase)
        audio_file = open('Agent_reply.wav', 'rb')
        update.message.reply_audio(audio_file)
        update.message.reply_text(phrase,reply_markup=ReplyKeyboardRemove())
        reply_keyboard = User['keyboard']
        update.message.reply_text('[Press \'Yes\' or \' No\' to respond]', reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))

    Agent['phrase'] = Action_dict['query_exe']['phrase_positive']
    return QUESTION_KEYBOARD


def agent_confirm_keyboard(update: Update, context: CallbackContext) -> int:
    global Agent, User, Global_objects, Global_variables
    if Agent['phrase'] != None:
        phrase = "\"" + str(Agent['phrase']) + "\""
        print(phrase)
        speak_simple(phrase)
        audio_file = open('Agent_reply.wav', 'rb')
        update.message.reply_audio(audio_file)
        update.message.reply_text(phrase,reply_markup=ReplyKeyboardRemove())
        reply_keyboard = User['keyboard']
        update.message.reply_text('[Press \'Yes\' or \' No\' to respond]', reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))

    Agent['phrase'] = Action_dict['query_exe']['phrase_positive']
    return CONFIRM_KEYBOARD


def video_note(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("Video received")
    video_file = update.message.video_note.get_file()
    video_file.download('video.mp4')
    print("Photo of {}: {}".format( user.first_name, '_audio.mp3'))
    update.message.reply_text(
        'What do you that I can make with your audio?'
    )

    reply_keyboard = [['Speech2Text', 'ExtractWords', 'Nothing']]
    update.message.reply_text(
        'Please, use the below buttons to choose my next action',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
    )

    return RESPONSE_TO_VOICE


### SIMPLE ACTION 
def speak_simple(text):
    tts = gTTS(text=text, lang='en')
    tts.save("Agent_reply.wav")
    print('\033[92mAgent-bot say: '+text+'\033[0m')
    playsound.playsound('Agent_reply.wav', True)


########################################################
#####    AGENT WAIT FOR A TEXT/VOCAL COMMAND
########################################################
### SIMPLE ACTION 


def read_usertext(update: Update, context: CallbackContext) -> int:
    global Agent, User
    user = update.message.from_user
    print("{} has sended following message : {}".format(user.first_name, update.message.text))
    
    if validators.url(update.message.text):
        return url_link(update.message.text, update, context)
    else:
        action, text = decode_text(update.message.text)
        if Agent['intent'] == None:
            if action == 'unknown':
                agent_phrase = Action_dict[str(text)]['phrase_positive'] + Action_dict['query_exe']['phrase_positive']
                phrase = "\"" + agent_phrase + "\""
                speak_simple(phrase)
                audio_file = open('Agent_reply.wav', 'rb')
                update.message.reply_audio(audio_file)
                update.message.reply_text(phrase,reply_markup=ReplyKeyboardRemove())
                return HEAR_USER
            elif action != 'unknown':
                return agent_main(input_main = action, type_main =  {'input': 'intent', 'type': 'text'}, update = update, context = context)
        else:
            return agent_main(input_main = str(text), type_main = {'input': 'objects', 'type': 'text'}, update = update, context = context)



    # # url = "https://www.youtube.com/watch?v=il_t1WVLNxk&list=PLqM7alHXFySGqCvcwfqqMrteqWukz9ZoE"
    # # video = pafy.new(url)
    # # video_file = video.streams[0]
    # #video_file = ffmpeg_streaming.input(best)
    # video_file = open('video.mp4', 'rb')
    # update.message.reply_video(video=video_file, supports_streaming=True)



def hear_uservoice(update: Update, context: CallbackContext) -> int:
    global Agent, User
    user = update.message.from_user
    audio_file = update.message.voice.get_file()
    audio_file.download('audio.wav')

    x,_ = librosa.load('audio.wav', sr=16000)
    sf.write('tmp.wav', x, 16000)
    r = sr.Recognizer()
    with sr.AudioFile('tmp.wav') as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        #comando = r.recognize_google(audio_file, language="en-EN")
        comando = s
    except Exception as e:
        print("Exception: "+str(e))


    action, text = decode_text(comando)
    if Agent['intent'] == None:
        if action == 'unknown':
            agent_phrase = Action_dict[str(text)]['phrase_positive'] + Action_dict['query_exe']['phrase_positive']
            phrase = "\"" + agent_phrase + "\""
            speak_simple(phrase)
            audio_file = open('Agent_reply.wav', 'rb')
            update.message.reply_audio(audio_file)
            update.message.reply_text(phrase,reply_markup=ReplyKeyboardRemove())
            return HEAR_USER
        elif action != 'unknown':
            return agent_main(input_main = action, type_main =  {'input': 'intent', 'type': 'text'}, update = update, context = context)
    else:
        return agent_main(input_main = str(text), type_main = {'input': 'objects', 'type': 'text'}, update = update, context = context)



def url_link(text, update, context):
    global Agent, User, Global_objects, Global_variables
    user = update.message.from_user
    Agent['prev_state'] = Agent['state']

    print('SONO ARIVATO QUI!!!')
    address_url = text
    if Global_variables['last_received'] == 'url_link':
        Global_variables['objects_receiving'] = Global_variables['objects_receiving'] + 1
    else:
        Global_variables['objects_receiving'] = 1
        Global_objects['url_link'] = []
        Global_variables['last_received'] = 'url_link'
    

    print("{} has sended following message : {}".format(user.first_name, address_url))
    
    if Agent['state'] == 'finding_parameters':
        key = Agent['state_parameter']['variable']
        list_objects = Action_dict[Agent['intent']]['params'][key]
        if list_objects == None:
            list_objects = []
        list_objects.append(address_url)
        Action_dict[Agent['intent']]['params'][key] = list_objects
        Agent['state_parameter']['step'] = None
        Agent['state_parameter']['variable'] = None
    else:
        Global_objects['url_link'].append(address_url)
        Agent['state'] = 'receiving_url_links'


    Agent['phrase'] = Params_dict['objects']['received'] + 'your url_link(s). '
        
    if (Agent['prev_state'] != Agent['state']):
        Agent['phrase'] = Agent['phrase'] + Action_dict['query_exe']['phrase_positive']
    elif Agent['prev_state'] == Agent['state']:
        Agent['phrase'] = None
    
    text = address_url
    return agent_main(input_main= text, type_main = {'input': 'objects', 'type': 'url_link'}, update = update, context = context)



def photo(update: Update, context: CallbackContext) -> int:
    global Agent, User, Global_objects, Global_variables
    user = update.message.from_user
    Agent['prev_state'] = Agent['state']


    if Global_variables['last_received'] == 'photo':
        Global_variables['objects_receiving'] = Global_variables['objects_receiving'] + 1
    else:
        Global_variables['objects_receiving'] = 1
        Global_objects['photo'] = []
        Global_variables['last_received'] = 'photo'
    
    photo_file = update.message.photo[-1].get_file()
    ts = str(int(time.time()))
    address_url = DOWNLOAD_DIR + 'user_photo-'+ts+str(Global_variables['objects_receiving'])+'.jpg'
    photo_file.download(address_url)
    print("{} has sended photo: {}".format( user.first_name, address_url))
    
    if Agent['state'] == 'finding_parameters':
        key = Agent['state_parameter']['variable']
        list_objects = Action_dict[Agent['intent']]['params'][key]
        if list_objects == None:
            list_objects = []
        list_objects.append(address_url)
        Action_dict[Agent['intent']]['params'][key] = list_objects
        Agent['state_parameter']['step'] = None
        Agent['state_parameter']['variable'] = None
    else:
        Global_objects['photo'].append(address_url)
        Agent['state'] = 'receiving_photos'


    Agent['phrase'] = Params_dict['objects']['received'] + 'your photo(s). '
        
    if (Agent['prev_state'] != Agent['state']):
        Agent['phrase'] = Agent['phrase'] + Action_dict['query_exe']['phrase_positive']
    elif Agent['prev_state'] == Agent['state']:
        Agent['phrase'] = None
    
    text = address_url
    return agent_main(input_main= text, type_main = {'input': 'objects', 'type': 'photo'}, update = update, context = context)





def video(update: Update, context: CallbackContext) -> int:
    global Agent, User, Global_objects, Global_variables
    user = update.message.from_user
    Agent['prev_state'] = Agent['state']


    if Global_variables['last_received'] == 'video':
        Global_variables['objects_receiving'] = Global_variables['objects_receiving'] + 1
    else:
        Global_variables['objects_receiving'] = 1
        Global_objects['video'] = []
        Global_variables['last_received'] = 'video'

    video_file = update.message.video.get_file()
    ts = str(int(time.time()))
    address_url = DOWNLOAD_DIR + 'user_video-'+ts+str(Global_variables['objects_receiving'])+'.jpg'
    video_file.download(address_url)
    print("{} has sended video: {}".format( user.first_name, address_url))
    
    if Agent['state'] == 'finding_parameters':
        key = Agent['state_parameter']['variable']
        list_objects = Action_dict[Agent['intent']]['params'][key]
        if list_objects == None:
            list_objects = []
        list_objects.append(address_url)
        Action_dict[Agent['intent']]['params'][key] = list_objects
        Agent['state_parameter']['step'] = None
        Agent['state_parameter']['variable'] = None
    else:
        Global_objects['video'].append(address_url)
        Agent['state'] = 'receiving_videos'


    Agent['phrase'] = Params_dict['objects']['received'] + 'your video(s). '
        
    if (Agent['prev_state'] != Agent['state']):
        Agent['phrase'] = Agent['phrase'] + Action_dict['query_exe']['phrase_positive']
    elif Agent['prev_state'] == Agent['state']:
        Agent['phrase'] = None
    
    text = address_url
    return agent_main(input_main= text, type_main = {'input': 'objects', 'type': 'video'}, update = update, context = context)



def video_note(update: Update, context: CallbackContext) -> int:
    global Agent, User, Global_objects, Global_variables
    user = update.message.from_user
    Agent['prev_state'] = Agent['state']


    if Global_variables['last_received'] == 'videonote':
        Global_variables['objects_receiving'] = Global_variables['objects_receiving'] + 1
    else:
        Global_variables['objects_receiving'] = 1
        Global_objects['videonote'] = []
        Global_variables['last_received'] = 'videonote'
    


    videonote_file = update.message.video_note.get_file()
    ts = str(int(time.time()))
    address_url = DOWNLOAD_DIR + 'user_videonote-'+ts+str(Global_variables['objects_receiving'])+'.jpg'
    videonote_file.download(address_url)
    print("{} has sended video_note: {}".format( user.first_name, address_url))
    
    if Agent['state'] == 'finding_parameters':
        key = Agent['state_parameter']['variable']
        list_objects = Action_dict[Agent['intent']]['params'][key]
        if list_objects == None:
            list_objects = []
        list_objects.append(address_url)
        Action_dict[Agent['intent']]['params'][key] = list_objects
        Agent['state_parameter']['step'] = None
        Agent['state_parameter']['variable'] = None
    else:
        Global_objects['videonote'].append(address_url)
        Agent['state'] = 'receiving_videonotes'


    Agent['phrase'] = Params_dict['objects']['received'] + 'your videonote(s). '
        
    if (Agent['prev_state'] != Agent['state']):
        Agent['phrase'] = Agent['phrase'] + Action_dict['query_exe']['phrase_positive']
    elif Agent['prev_state'] == Agent['state']:
        Agent['phrase'] = None
    
    text = address_url
    return agent_main(input_main= text, type_main = {'input': 'objects', 'type': 'videonote'}, update = update, context = context)



def keyboard(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("{} has sended following message : {}".format(user.first_name, update.message.text))

    confirm = decode_bool_text(update.message.text)
    return agent_main(input_main = confirm, type_main = {'input': 'confirmation', 'type': 'boolean'}, update = update, context = context)



########################################################
#####    AGENT ANALYZES INTENTS
########################################################
### SIMPLE ACTION 
def decode_text(doc, debug_dependencies=True):
    text = nlp(doc)
    action = extract_from_Intent_dataset(text)
    if debug_dependencies:
        print('************ Phrase Analysis  ************')
        print('Print dependencies')
        text.sentences[0].print_dependencies()
        # print('Print tokens')
        # text.sentences[0].print_tokens()
    return action, text.text


def extract_from_Intent_dataset(doc):
    action = 'unknown'
    for sent in doc.sentences:
        for wrd in sent.words:
            print('word analyzer: ', wrd)
            if wrd.lemma in Action_dict.keys():
                #action_text = Action_dict[wrd.lemma]
                action = wrd.lemma
    return action

def decode_bool_text(doc, debug_dependencies=False):
    if doc == 'Yes':
        return True
    else:
        return False


########################################################
#####    AGENT EXECUTES INTENTS
########################################################
### SIMPLE ACTION 
def agent_main(input_main, type_main, update: Update, context: CallbackContext) -> int:
    global Agent, User, Global_objects, Global_variables

    #prendi tutto il dictionary di saving
    # verifica di avere tutti i parametri
    if (type_main['input'] == 'intent'):
        print('***  INPUT RECEIVED =  INTENT  ***')
        Action = Action_dict[input_main]  # this is a dictionary relative to the intent eg "{'params': {'objects': '', 'tags': ''}, 'exe': 'saving_document', 'phrase_positive': "Let's start Saving function", 'description': 'This function will save docu/photo/link and whatever else, giving them a tag. You can recover them, asking to system'}"
        Agent['intent'] = input_main      # this is the name of the intent eg "save"
        Agent['state'] = 'finding_parameters'
        params = Action['params']
    elif (type_main['input'] == 'objects'):
        print('***  INPUT RECEIVED =  OBJECTS  ***')
        if Agent['intent'] != None:
            Action = Action_dict[Agent['intent']]
            params = Action['params']
        else:
            agent_action(update, context)
    elif (type_main['input'] == 'confirmation'):
        print('***  INPUT RECEIVED =  CONFIRMATION  ***')
        if Agent['intent'] != None:
            Action = Action_dict[Agent['intent']]
            params = Action['params']
        else:
            agent_action(update, context)
        response = input_main

    
    if Agent['state'] == 'finding_parameters':
        # Agent['prev_state'] = Agent['state']
        # Agent['state'] = 'checking_globalvar'
        if params != None:
            for key in params.keys():
                if Action_dict[Agent['intent']]['params'][key] == None:
                    if (key == 'objects') and check_globalvar(key) in Global_objects:
                        if (Agent['state_parameter']['step'] == 'confirmation') and (type_main['input'] == 'confirmation'):
                            type_parameter = Agent['state_parameter']['variable']
                            Agent['state_parameter']['step'] = None
                            Agent['state_parameter']['variable'] = None
                            if response:
                                Action_dict[Agent['intent']]['params'][key] = Global_objects[type_parameter]
                                #print('TESTED THIS FINAL 1')
                            else:
                                Global_objects[type_parameter] = []                 
                                Agent['phrase'] = Params_dict[key]['request'] 
                                Agent['state_parameter']['step'] = 'asking_parameters'
                                Agent['state_parameter']['variable'] = key
                                #print('TESTED THIS FINAL 8')
                                return agent_action(update, context)
                        else:
                            #print('TESTED THIS FINAL 2')
                            return agent_confirm_keyboard(update, context)

                    elif (key == 'objects') and check_globalvar(key) == 'others':
                        print('NOT TESTED FINAL')
                    else:
                        if (Agent['state_parameter']['step'] == 'asking_parameters') and (type_main['input'] == 'objects') and (type_main['type'] == 'text'):
                            z = input_main.strip()
                            Agent['phrase'] = '['+ str(z) + '] '+ str(Params_dict[key]['ask_confirmation'])
                            User['keyboard'] = Keyboard_dict['boolean'][0]
                            User['keyboard_regex'] = Keyboard_dict['boolean'][1]
                            Agent['state_parameter']['step'] = 'confirmation'
                            Agent['state_parameter']['variable'] = z
                            #print('TESTED THIS FINAL 4')
                            return agent_confirm_keyboard(update, context)

                        elif (Agent['state_parameter']['step'] == 'asking_parameters') and (type_main['input'] == 'objects') and (type_main['type'] in Objects_type):
                            z = input_main.strip()
                            Agent['phrase'] = '['+ str(z) + '] '+ str(Params_dict[key]['ask_confirmation'])
                            User['keyboard'] = Keyboard_dict['boolean'][0]
                            User['keyboard_regex'] = Keyboard_dict['boolean'][1]
                            Agent['state_parameter']['step'] = 'confirmation'
                            Agent['state_parameter']['variable'] = z
                            #print('TESTED THIS FINAL 9')
                            return agent_confirm_keyboard(update, context)

                        elif (Agent['state_parameter']['step'] == 'confirmation') and (type_main['input'] == 'confirmation'):
                            if response:
                                type_parameter = Agent['state_parameter']['variable']
                                Agent['state_parameter']['step'] = None
                                Agent['state_parameter']['variable'] = None
                                Action_dict[Agent['intent']]['params'][key] = type_parameter
                                #print('TESTED THIS FINAL 5')
                            else:
                                print('NOT TESTED FINAL')
                        else:
                            Agent['phrase'] = Params_dict[key]['request'] 
                            Agent['state_parameter']['step'] = 'asking_parameters'
                            Agent['state_parameter']['variable'] = key
                            #print('TESTED THIS FINAL 3')
                            return agent_action(update, context)


                else:
                    if (Agent['state_parameter']['step'] == 'asking_parameters') and (type_main['input'] == 'objects') and (type_main['type'] in Objects_type):
                        z = input_main.strip()
                        Agent['phrase'] = '['+ str(z) + '] '+ str(Params_dict[key]['ask_confirmation'])
                        User['keyboard'] = Keyboard_dict['boolean'][0]
                        User['keyboard_regex'] = Keyboard_dict['boolean'][1]
                        Agent['state_parameter']['step'] = 'confirmation'
                        Agent['state_parameter']['variable'] = z
                        #print('TESTED THIS FINAL 9')
                        return agent_confirm_keyboard(update, context)


        Agent['state'] = 'executing_action'
        #print('TESTED THIS FINAL 6')

    if Agent['state'] == 'executing_action':
        #fun_aux.create_dictio(ARCHIVE_DIR)
        if Agent['intent'] == 'save':
            input_object = Action_dict[Agent['intent']]['params']['objects']
            type_object = Action_dict[Agent['intent']]['params']['tags']
            fun_aux.saving_function(ARCHIVE_DIR, input_object, type_object)

        elif Agent['intent'] == 'read':
            type_object = Action_dict[Agent['intent']]['params']['tags']
            object_retrived = fun_aux.reading_function(ARCHIVE_DIR, type_object)
            agent_show(object_retrived, update, context)

        elif Agent['intent'] == 'send':
            object_retrived = fun_aux.ipaddress_function()
            agent_show(object_retrived, update, context)
        Agent['phrase'] = Action_dict[Agent['intent']]['phrase_positive']

        Action_resetstate(Action_dict, Agent)
        Agent_resetstate(Agent)
        return agent_action(update, context)



def Action_resetstate(action, agent):
    params = action[agent['intent']]['params']
    if params != None:
        for key in params.keys():
            action[agent['intent']]['params'][key] = None


def Agent_resetstate(agent):
    agent['intent'] = None
    agent['state'] = None
    agent['state_parameter']['step'] = None
    agent['state_parameter']['variable'] = None

def check_globalvar(key):
    global Agent, User, Global_objects, Global_variables, Objects_type
    objects_found = {}
    last_received = ''
    received_name = []
    for type_obj in Objects_type:        
        if len(Global_objects[type_obj]) != 0:
            objects_found[type_obj] = len(Global_objects[type_obj])
            received_name.append(type_obj)
            if Global_variables['last_received'] == type_obj:
                last_received = type_obj
    

    if len(objects_found) == 0:
        return False
    elif len(objects_found) == 1:
        Agent['phrase'] = Params_dict[key]['found_in_globalvariable'] + str(objects_found[last_received]) + ' ' + str(last_received) + ' in input. Are they?'
        User['keyboard'] = Keyboard_dict['boolean'][0]
        User['keyboard_regex'] = Keyboard_dict['boolean'][1]
        Agent['state_parameter']['step'] = 'confirmation'
        Agent['state_parameter']['variable'] = last_received
        return type_obj
    elif len(objects_found) > 1:
        Agent['phrase'] = Params_dict[key]['found_in_globalvariable'] 
        User['keyboard'] = Keyboard_dict['check_alternatives'][0]
        User['keyboard_regex'] = Keyboard_dict['check_alternatives'][1]
        Agent['state_parameter']['step'] = 'confirmation'
        Agent['state_parameter']['variable'] = received_name
        return 'others'
    else:
        Agent['phrase'] = Params_dict[key]['request']
        return 3



def utterance_analysis(doc):
    parsed_text = {'word':[], 'pos':[], 'lemma':[], 'dependency_relation':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            if wrd.pos in VB_dict.keys():
                parsed_text['word'].append(wrd.text)
                parsed_text['pos'].append(wrd.pos)
                parsed_text['lemma'].append(wrd.lemma)
                parsed_text['dependency_relation'].append(wrd.dependency_relation)
    return parsed_text



def hear():
    r = sr.Recognizer()
    #r.energy_threshold = 200
    
    err = True
    while err:
        input("[Talk after pushing ENTER]")

        with sr.Microphone() as source:
            print("[AGENT_REPLY listening...]")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            print("[... End listening]")
            try:
                err = False
                text = r.recognize_google(audio, language="it-IT")
            except sr.UnknownValueError:
                err = True#print("Google Cloud Speech could not understand audio")
                print('Alza la voce per favore')
            except sr.RequestError as e:
                err = True#print("Could not request results from Google Cloud Speech service; {0}".format(e))
                print('Alza la voce per favore')
    print('\033[94mTu: '+text.lower()+'\033[0m')
    return text.lower()




def response_to_photo(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("User has sended following message {}: {}".format( user.first_name, update.message.text))
    update.message.reply_text(
        'I\'m analyzing your message'
        'Now we will go in HEAR_USER'
        ' from response to photo',
        reply_markup=ReplyKeyboardRemove(),
    )

    return HEAR_USER


def response_to_voice(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("User has sended following message {}: {}".format( user.first_name, update.message.text))
    update.message.reply_text(
        'I\'m analyzing your message'
        'Now we will go in HEAR_USER'
        ' from response to photo',
        reply_markup=ReplyKeyboardRemove(),
    )

    return HEAR_USER
   


def skip_audiophoto(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("User {} did not send a photo.".format( user.first_name))
    update.message.reply_text(
        'I bet you look great! Now, send me your location please, ' 'or send /skip.'
    )

    return HEAR_USER




def location(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    user_location = update.message.location
    print(
        "Location of {}: %f / %f".format( user.first_name, user_location.latitude, user_location.longitude
    ))
    update.message.reply_text(
        'Maybe I can visit you sometime! ' 'At last, tell me something about yourself.'
    )

    return BIO


def skip_location(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("User {} did not send a location.".format( user.first_name))
    update.message.reply_text(
        'You seem a bit paranoid! ' 'At last, tell me something about yourself.'
    )

    return BIO


def bio(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("Bio of {}: {}".format( user.first_name, update.message.text))
    update.message.reply_text('Thank you! I hope we can talk again some day.')

    return ConversationHandler.END


def cancel(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    print("User {} canceled the conversation.".format( user.first_name))
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END



def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)



#viene stampata il tavolo da gioco in cui composto da 100 o più caselle
def print_board(agent_position):
    fields = list(range(100))

    print("[A = Agent] [C = Casa] [S = Serra] [O = Orto] [G = Giardino]...")
    board = "-----------------------------------------\n"
    for i in range(0, 100, 10):
        delimiter = fields[i:i+10]
        for field in delimiter:
            if field == agent_position:
                board += "| A "
            elif field == casa1 or field == casa2 or field == casa3 or field == casa4 or field == casa5 or field == casa6 or field == casa7:
                board += "| C "
            elif field == serra0 or field == serra1 or field == serra2 or field == serra3 or field == serra4 or field == serra5:
                board += "| S "
            elif field == orto0 or field == orto1 or field == orto2 or field == orto3 or field == orto4 or field == orto5:
                board += "| O "
            elif field == giardino0 or field == giardino1 or field == giardino2 or field == giardino3 or field == giardino4 or field == giardino5:
                board += "| G "
            elif field == field == rose1 or field == rose2:
                board += "| R "
            elif field == field == bush1 or field == bush2:
                board += "| B "
            elif field == field == pomodoro1 or field == pomodoro2:
                board += "| P "
            elif field == field == forbici:
                board += "| F "
            elif field == field == irrigatore:
                board += "| I "
            elif field == field == semi:
                board += "| N "

            else:
                board += "|   "
        board += "|\n"
        board += "-----------------------------------------\n"     
    print(board)


def main() -> None:
    global Agent, User, Global_objects, Global_variables
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    print('**************************************************************')
    print('                  CHATBOT SESSION START                       ')
    print('**************************************************************')
    updater = Updater("1522170282:AAH6lnHKhcchjL_0lfPGxromcP3hE3QvFMA".format(use_context=True))

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states GENDER, PHOTO, LOCATION and BIO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', agent_action)],
        states={
            AGENT_REPLY: [CommandHandler('reply', agent_action)],
            HEAR_USER: [MessageHandler(Filters.text, read_usertext),
            MessageHandler(Filters.photo, photo), CommandHandler('skip', skip_audiophoto), 
            MessageHandler(Filters.voice, hear_uservoice), CommandHandler('skip', skip_audiophoto),
            MessageHandler(Filters.video_note, video_note), CommandHandler('skip', skip_audiophoto),
            MessageHandler(Filters.video, video), CommandHandler('skip', skip_audiophoto),
            MessageHandler(Filters.location, location),],
            CONFIRM_KEYBOARD: [MessageHandler(Filters.regex('^(Yes|No)$'), keyboard)],
            QUESTION_KEYBOARD: [MessageHandler(Filters.regex(str(User['keyboard_regex'])), keyboard)],
            RESPONSE_TO_PHOTO: [MessageHandler(Filters.text, response_to_photo)],
            RESPONSE_TO_VOICE: [MessageHandler(Filters.text, response_to_voice)],
            LOCATION: [MessageHandler(Filters.location, location),CommandHandler('skip', skip_location),],
            BIO: [MessageHandler(Filters.text & ~Filters.command, bio)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry = True,
    )

    dispatcher.add_handler(conv_handler)

    # AGENT_REPLY the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # AGENT_REPLY_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()



if __name__ == '__main__':
    main()
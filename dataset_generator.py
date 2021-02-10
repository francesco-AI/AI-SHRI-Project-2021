import random

TRAINSET_DIR = 'trainset.json'
TESTSET_DIR = 'testset.json'


phrase_elements = {
'greetings' : ['', 'Please'],
'subject' : ['', 'I', 'You', 'he', 'she', 'it', 'they', 'we', 'Lucy', 'John', 'Mary', 'Steve', 'Sarah', 'Let\'s'],
'want' : ['', 'want to', 'would to', 'are trying to', 'prepare for', 'desire to', 'hope to', 'try to', 'work for'],
'type_of_action' : ['save', 'read'],
'action' : {'save': ['save', 'conserve', 'mantain', 'preserve', 'shield', 'keep safe', 'keep up', 'safeguard', 'defend', 'take care of', 'shield'],
'read': ['load', 'read', 'recover', 'get claim', 'restore', 'resume', 'retrieve', 'get back', 'reclaim', 'recoup', 'regain', 'regain', 'rescue', 'reposses', 'redeem', 'bring back', 'catch up', 'retake', 'obtain', 'reacquire', 'take back']},
'quantity': ['', 'one', 'some', 'many', 'two', 'multiple', 'not so much'],
'object_doc' : ['document', 'photo', 'file', 'link', 'url', 'video', 'videonote', 'object', 'element', 'file'],
'where' : ['from dataset', 'from storage', 'from archive', 'from drive', 'from memory'], 
'greet_final' : ['Thanks', 'Thank you', 'please', '',],
}

phrase_elements_index = {0: 'greetings', 1: 'subject', 2: 'want', 3: 'type_of_action', 4: 'action', 5: 'quantity', 6: 'object_doc', 7: 'where', 8: 'greet_final'}



def composer_phrase(phrase_elements, index):
    phrase = ''
    for i in range(len(phrase_elements) ):
        if i != 3:
            if i == 4:
                array_elements = phrase_elements[index[i]][label]
            else:
                array_elements = phrase_elements[index[i]]

            rand_element = random.randint(0, len(array_elements)-1)
            element = array_elements[rand_element]
            phrase = phrase + element + ' '
        else:
            array_elements = phrase_elements[index[i]]
            rand_element = random.randint(0, len(array_elements)-1)
            label = array_elements[rand_element] 
    dataset_element = '{"text": "' + str(phrase) + '", "language": "EN", "label": "' + str(label) + '"}' + '\n'


    return dataset_element



def saving_phrase(file_archive, phrase):
    f=open(file_archive, "a+")
    f.write(phrase)
    f.close()

def clear_file(file_archive):
    open(file_archive, 'w').close()

# for i in range(100):
#     array_elements = phrase_elements[phrase_elements_index[7]]
#     rand_element = random.randint(0, len(array_elements)-1)
#     print(array_elements[rand_element])

# phrase_final = composer_phrase(phrase_elements, phrase_elements_index)
# print('FINAL PHRASE IS : ', phrase_final)


clear_file(TRAINSET_DIR)
clear_file(TESTSET_DIR)
num_element = 100 * 10
for i in range (num_element):
    if i < (num_element / 10):
        phrase_final = composer_phrase(phrase_elements, phrase_elements_index)
        saving_phrase(TESTSET_DIR, phrase_final)
    else:
        phrase_final = composer_phrase(phrase_elements, phrase_elements_index)
        saving_phrase(TRAINSET_DIR, phrase_final)
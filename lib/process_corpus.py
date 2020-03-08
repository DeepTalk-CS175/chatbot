def process_data(f1="./train/dialogues_train.txt", 
    f2="./train/dialogues_act_train.txt", 
    f3="./train/dialogues_emotion_train.txt", num_prev = -2):

    utterances = []
    next_utterance = []
    previous_utterances = []
    index = 0

    for i, dialog in enumerate(open(f1, mode='r',encoding='utf-8')):
        sentences = dialog.rstrip('\n __eou__').split('__eou__')  # sentences of a dialog
        index_start = index
        for line in sentences:
            utterances.append(line)
            next_utterance.append(index + 1)
            prev_list = [i for i in range(index_start, index)]
            index += 1
            previous_utterances.append(prev_list[num_prev:])
        # last utterance in a dialog has no "next utterance" so set it to -1
        next_utterance[-1] = -1

    # get all act labels
    all_acts = []
    for line in open(f2):
        acts = line.rstrip(' \n').split(' ')
        for act in acts:
            all_acts.append(act)

    # get all emotion labels
    all_emotions = []
    for line in open(f3):
        emotions = line.rstrip(' \n').split(' ')
        for emotion in emotions:
            all_emotions.append(emotion)

    assert len(utterances) > 0
    assert len(utterances) == len(next_utterance)
    assert len(utterances) == len(previous_utterances)
    assert len(utterances) == len(all_acts)
    assert len(utterances) == len(all_emotions)
    return utterances, next_utterance, previous_utterances, all_acts, all_emotions

if __name__ == '__main__':
    utterances, next_utterance, previous_utterances, actions, emotions = process_data()
    print("corpus size:",len(utterances))
    data_dict = []
    for i in range(0, 4):
        print("index:", i)
        print("sentence:", utterances[i])
        print("next:", next_utterance[i])
        print("previous:", previous_utterances[i])
        print("action:", actions[i])
        print("emotion:", emotions[i])
        print("\n")

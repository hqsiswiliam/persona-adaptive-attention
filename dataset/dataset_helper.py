import re


def read_personachat_split(split_dir):
    persona = []
    their_persona = []
    query = []
    response = []
    candidates = []
    their_per_group = None
    try:
        with open(split_dir, "r", encoding="utf-8") as src:
            pre_st, st = 'dia', 'dia'
            for line in src:
                line = line.strip()
                if 'your persona:' in line:
                    pre_st = st
                    st = 'per'
                elif 'partner\'s persona' in line:
                    pre_st = st
                    st = 'their_per'
                else:
                    pre_st = st
                    st = 'dia'
                if pre_st == 'dia' and st == 'per':
                    per_group = ''
                if pre_st == 'per' and st == 'their_per':
                    their_per_group = ''
                if st == 'per':
                    per_group += (line[16:] + "\t")
                elif st == 'their_per':
                    their_per_group += (line[21:] + ' ')
                elif st == 'dia':
                    persona.append(per_group)
                    line = line[line.find(' '):]
                    query.append(line.split('\t')[0])
                    response.append(line.split('\t')[1])
                    candidates.append(line.split('\t')[3].split('|'))
                    if their_per_group is not None:
                        their_persona.append(their_per_group)
                else:
                    raise (ValueError)
    except FileNotFoundError:
        print(f"Sorry! The file {split_dir} can't be found.")
    return persona, query, response, their_persona, candidates


def combine_persona_query_response(persona, query, response, candidates):
    assert ((len(persona) == len(query)) and (len(query) == len(response))), \
        'the length of persona, query, response must be equivalent'
    data = {}
    for index, psn in enumerate(persona):
        split_persona = psn.strip().split("\t")
        psn = psn.replace("\t", " ").strip()
        if psn not in data.keys():
            data[psn] = {'persona': psn, 'query': [], 'response': [], 'dialog': [], 'response_turns': 0,
                         'persona_list': split_persona, 'candidates': []}
        data[psn]['query'].append(query[index])
        data[psn]['response'].append(response[index])
        data[psn]['dialog'].append(query[index])
        data[psn]['dialog'].append(response[index])
        data[psn]['candidates'].append(candidates[index])
        data[psn]['response_turns'] += 1
    return data


def preprocess_text(text):
    punctuations = '.,?'
    for punc in punctuations:
        text = text.replace(punc, ' {} '.format(punc))
    text = re.sub(' +', ' ', text).strip()
    return text


def preprocess_texts(text_array):
    return [preprocess_text(t) for t in text_array]


def get_chat_by_turns(combined_data, turns=1,
                      sep_token='[SEP]', add_role_indicator=True,
                      add_persona_indicator=True, max_context_turns=-1):
    assert turns > 0, 'turns must be large than 0'
    all_persona = list(combined_data.keys())
    filtered_persona = list(filter(lambda p: combined_data[p]['response_turns'] >= turns, all_persona))
    data = []

    for single_persona in filtered_persona:
        single_persona_data = combined_data[single_persona]
        persona_list = single_persona_data['persona_list']
        context = []
        for index, (query, response) in enumerate(
                zip(single_persona_data['query'], single_persona_data['response'])
        ):
            if max_context_turns != -1 and \
                    index + 1 < single_persona_data['response_turns'] - max_context_turns:
                continue
            if add_role_indicator:
                query = "Q: {}".format(query)
                if not index + 1 >= turns:
                    response = "R: {}".format(response)
            context += [query, response]
            if index + 1 >= turns:
                break

        response = context[-1]
        context = context[:-1]

        input_x_str = " {} ".format(sep_token).join(context)
        input_x_str = re.sub(" +", " ", input_x_str)
        if add_persona_indicator:
            single_persona = "P: {}".format(single_persona)
        data.append({'input': preprocess_texts(context),
                     'input_str': preprocess_text(input_x_str),
                     'target': preprocess_text(response),
                     'persona': preprocess_text(single_persona),
                     'persona_list': preprocess_texts(persona_list),
                     'candidates': preprocess_texts(single_persona_data['candidates'][-1])})
    return data

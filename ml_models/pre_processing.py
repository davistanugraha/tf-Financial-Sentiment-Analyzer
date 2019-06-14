import pandas as pd
import re
import logging


def remove_greetings_filter(sentence):
    """
    checks if sentence contains greetings and returns True if it does
    :param sentence: string representing text that might represent a greeting
    :return: boolean with True meaning that the sentence is a greeting and False if it isn't
    """
    # potential greeting phrases that makes a sentence greeting.
    GREETINGS = ['thank you', 'good question',
                 'good morning', 'good afternoon', 'have a good day', 'good evening', 'have a great day',
                 'let me remind you of the caution regarding forward-looking', 'operator:', 'microphone',
                 'trouble hearing you']
    return_status = False
    # checks if any of the words in GREETINGS is in sentence
    for greeting in GREETINGS:
        if greeting in sentence.lower():
            return_status = True
            break
    # remove sentences if below length of 4.
    if len(sentence.split(' ')) < 6:
        return_status = True
    return return_status


def mds_closing_remarks_preprocess(earnings_doc_file):
    """
    pre-processing for management discussion section and closing remarks section.
    :param earnings_doc_file: dictionary/json object representing output from pdf parser.
    :return: a dictionary of dattaframes corresponding to earningsc all segmented by sentences
    and paragraphs
    """
    # extracting mds section per sentence level.
    mds_sentence_text = []
    # extracting mds section paragraph level
    mds_para_text = []
    # extracting closing remarks section per sentence level.
    cr_text = []
    # extracting closing remarks section paragraph level
    cr_para_text = []
    # iterate through sections in management discussion section.
    for sub_sections in earnings_doc_file['management_discussion_section']:
        # use regex to split paragraphs to sentences
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", sub_sections['Text'])
        # assign information to each sentence.
        sentence_info = [[sentence, sub_sections['Speaker'], sub_sections['Role'], earnings_doc_file['company']['name']]
                         for sentence in sentences]
        mds_sentence_text += sentence_info
        # assigns infomration to each paragraph
        mds_para_text += [
            [sub_sections['Text'], sub_sections['Speaker'], sub_sections['Role'], earnings_doc_file['company']['name']]]
    try:
        # breaking down closing remarks to sentence and paragraph level
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s",
                             earnings_doc_file['final_remarks']['Text'])
        sentence_info = [
            [sentence, earnings_doc_file['final_remarks']['Speaker'], earnings_doc_file['final_remarks']['Title'],
             earnings_doc_file['company']['name']] for sentence in sentences]
        cr_text += sentence_info
        cr_para_text += [[earnings_doc_file['final_remarks']['Text'], earnings_doc_file['final_remarks']['Speaker'],
                          earnings_doc_file['final_remarks']['Title'], earnings_doc_file['company']['name']]]
    except:
        logging.exception('final_remarks sections does not exist')
        pass
    # converting mds and cr 2d arrays to dataframe as per modelling data input type.    
    para_sentence_df = pd.DataFrame(mds_sentence_text, columns=['sentences', 'speaker', 'role', 'company'])
    para_paragraph_df = pd.DataFrame(mds_para_text, columns=['paragraphs', 'speaker', 'role', 'company'])
    cr_sentence_df = pd.DataFrame(cr_text, columns=['sentences', 'speaker', 'role', 'company'])
    cr_paragraph_df = pd.DataFrame(cr_para_text, columns=['paragraphs', 'speaker', 'role', 'company'])
    final_json = {'mds_sentence': para_sentence_df, 'mds_paragraph': para_paragraph_df, 'cr_sentence': cr_sentence_df,
                  'cr_paragraph': cr_paragraph_df}
    return final_json


def mds_cr_to_dict(mds_df, initial_index):
    """
    takes mds related output from mds_closing_remarks_preprocess with an extra column called "label" and 
    assigns sentiment related information in dictionary format.(need to be ingestible by database)
    :param mds_df: dataframe output of mds_closing_remarks_preprocess with an extra column called "label"
    :param initial_index: integer at which index should start counting from
    :return: a tuple with first element being a dictionary containing sentiment information to mds section
    second element of tuple being the last semtemce index of the mds section.
    """
    # gives greetings and pleasantries label of neutral
    mds_df['greeting'] = mds_df.sentences.apply(remove_greetings_filter)
    mds_df.loc[mds_df.greeting == True, 'label'] = 'neutral'
    # keep track of positive and negative sentiment sentence count
    positive_count = 0
    negative_count = 0
    contents = []
    # iterate through dataframe row-wise, extract information and converts it to dictionary format
    for index, row in mds_df.iterrows():
        text_info = {"text": row['sentences'], "sentiment": row['label'], "speaker": row['speaker'],
                     "speaker_title": row['role'], "company": row['company']}
        content = {"index": initial_index, "content": text_info}
        contents += [content]
        if row['label'] == 'positive':
            positive_count += 1
        elif row['label'] == 'negative':
            negative_count += 1
        # keeps track sentence index.    
        initial_index += 1
    # assigns 1 to positive and negative count to avoid division by 0.    
    if positive_count == 0 and negative_count == 0:
        positive_count = 1
        negative_count = 1
    # scaling it from -1 to 1
    mds_sentiment_score = (positive_count / (positive_count + negative_count)) * 2 - 1
    final_tuple = ({"sentiment_score": mds_sentiment_score, "contents": contents}, initial_index)
    return final_tuple


def qna_preprocessing_section(json_segment, company_name):
    """
    process qna section of pdf parser output and segment it per sentence or paragraoph.
    :param json_segment: dictionary value corresponding to qna section of pdf parse output
    :param company_name: company_name of transcript we're analyzing(will be obtained from pdf_parse output)
    :return: dictionary with values being dataframe of qna segmented by sentences or paragraph.
    """
    # extracting qna section per sentence level
    qna_sentence_block = []
    # extracting qna section per paragraph level
    qna_para_block = []
    qna_para_block += [[json_segment['Question_text'], 'Q', json_segment['Q_speaker'], json_segment['Q_title'],
                        json_segment['Q_firm']]]
    # splitting paragraph to sentences by regex.
    question_text = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", json_segment['Question_text'])
    # assigns infomration to each sentence
    question_texts_info = [[sentence, 'Q', json_segment['Q_speaker'], json_segment['Q_title'], json_segment['Q_firm']]
                           for sentence in question_text]
    qna_sentence_block += question_texts_info
    # iterate through answers in that specific question and answer block.
    for a_sections in json_segment['Answer']:
        # extract information from answer paragraph level
        qna_para_block += [
            [a_sections['A_text'], 'A', a_sections['A_speaker'], a_sections['A_title'], json_segment['Q_firm']]]
        # split answer tp sentences.
        answer_text = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", a_sections['A_text'])
        # extract information from answer sentence level.
        answer_text_array = [[sentence, 'A', a_sections['A_speaker'], a_sections['A_title'], company_name] for sentence
                             in answer_text]
        qna_sentence_block += answer_text_array
    # converting mds and cr 2d arrays to dataframe as per modelling data input type.        
    qna_sentence_df = pd.DataFrame(qna_sentence_block, columns=['sentences', 'type', 'speaker', 'role', 'company'])
    qna_paragraph_df = pd.DataFrame(qna_para_block, columns=['paragraphs', 'type', 'speaker', 'role', 'company'])
    final_json = {'sentences': qna_sentence_df, 'paragraph': qna_paragraph_df}
    return final_json


def qna_to_dict(qna_df, index_tracker):
    """
    takes question and answers related output from qna_preprocessing_section with an extra column called "label" and 
    assigns sentiment related information. (we're processing one qna block not the entire qna section)
    :param qna_df: dataframe output of qna_preprocessing_section with an extra column called "label"
    :param index_tracker: integer at which index should start counting from
    :return: set with first element being a dictionary containing question and answer sentiment information
    second element being the last index of the last sentence, third element being a dictionary keeping numerical information
    on sentiment count.
    """
    # gives greetings and pleasantries label of neutral
    qna_df['greeting'] = qna_df.sentences.apply(remove_greetings_filter)
    qna_df.loc[qna_df.greeting == True, 'label'] = 'neutral'
    # keep track of positive and negative sentiment sentence count
    question_positive_count = 0
    question_negative_count = 0
    answers_positive_count = 0
    answers_negative_count = 0
    content = {}
    questions = []
    answers = []
    # iterate through dataframe row-wise, extract information and converts it to dictionary format
    for index, row in qna_df.iterrows():
        # extract info of every column for that row.
        text_info = {"text": row['sentences'], "sentiment": row['label'], "speaker": row['speaker'],
                     "speaker_title": row['role'], "company": row['company']}
        # assign text info as a value to content.
        content = {"index": index_tracker, "content": text_info}
        # update counts on type of text as iterating through rows.
        if row['type'] == 'Q':
            questions += [content]
            if row['label'] == 'positive':
                question_positive_count += 1
            elif row['label'] == 'negative':
                question_negative_count += 1
            index_tracker += 1
        else:
            answers += [content]
            if row['label'] == 'positive':
                answers_positive_count += 1
            elif row['label'] == 'negative':
                answers_negative_count += 1
            index_tracker += 1
    final_set = ({"question": questions, "answer": answers}, index_tracker,
                 {'q_pos_count': question_positive_count, 'q_neg_count': question_negative_count,
                  'a_pos_count': answers_positive_count, 'a_neg_count': answers_negative_count})
    return final_set

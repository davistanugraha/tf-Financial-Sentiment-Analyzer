import pandas as pd
import numpy as np
import re
from re import finditer
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from io import StringIO
import json
import logging


# CoverPage


def coverpageinfo(earnings_call_file):
    """
    takes in a pdf file and returns a dictionary with information regarding the cover page(company_name,symbol,date,year)
    :param earnings_call_file: File - representation of the earnings call file
    :return: dictionary with keys corresponding to the pdf basic info such as 'symbol','quarter_year','company_name' and 'published date'
    """
    coverpage_info = {}
    PASSWORD = ""
    # setting parameters for pdfminer's get_pages function
    MAXPAGES = 0
    CACHING = True
    PAGE_NUMBERS = set()
    # using loop count to count until first page in order to only extract cover page
    loop_count = 0
    text = ""
    for page in PDFPage.get_pages(earnings_call_file, PAGE_NUMBERS, maxpages=MAXPAGES, password=PASSWORD,
                                  caching=CACHING, check_extractable=True):
        # setting parameters for PDFMiner's TextConverter function
        resource_manager = PDFResourceManager()
        return_string = StringIO()
        CODENCODING = 'utf-8'
        analysis_parameter = LAParams()
        device = TextConverter(resource_manager, return_string,
                               codec=CODENCODING, laparams=analysis_parameter)
        # updates interpreter with current page
        interpreter = PDFPageInterpreter(resource_manager, device)
        interpreter.process_page(page)
        # stop at 0 as function only extract title page,loop breaks after going through cover page
        if loop_count == 0:
            text = return_string.getvalue()
            break
    device.close()
    return_string.close()
    # removing footer from page
    cleansed_text = re.search(
        "(?<= FactSet CallStreet, LLC).*", text).group(0).strip()
    # use regex to match and extract company ticker
    ticker = re.search("\((.*?)\)", cleansed_text).group(0).strip()
    # extracting company ticker symbol
    splits = cleansed_text.split(ticker)
    # using ticker as a regex input to identify quarter and year of pdf
    splits = [i.strip() for i in splits]
    # extracting quarter_year information as it exist and reolacing white spaces(numerical integers are
    # used as ingested data is assumed to be structured)
    quarter_year = splits[1][:7].replace(" ", "_")
    # using ticker as a regex input to identify published date of pdf
    published_date_time = splits[0][:11]
    # using ticker as a regex input to identify company namne
    company_name = splits[0][12:]
    # putting all the information of the cover page in a dictionary.
    coverpage_info['symbol'] = ticker[1:-1]
    coverpage_info['quarter_year'] = quarter_year
    coverpage_info['company_name'] = company_name
    coverpage_info['published_date'] = published_date_time
    return coverpage_info


# Preprocessing


def append_pdf_pages(earnings_call_file, exclude_page):
    """
    append pages of the pdf and also removing header,footers,line breaks and disclaimer section.
    :param earnings_call_file: File - representation of the earnings call file
    :param exclude_page: list of numbers indicating pages you want to exclude for appending
    :return: string in pdf with all linebreaks, headers,footers and disclaimer section removed.
    """
    PASSWORD = ""
    MAXPAGES = 0
    CACHING = True
    pdf_string = ""
    PAGE_NUMBERS = set()
    loop_count = 0
    for page in PDFPage.get_pages(earnings_call_file, PAGE_NUMBERS, maxpages=MAXPAGES, password=PASSWORD,
                                  caching=CACHING,
                                  check_extractable=True):
        # skips through pages which we dont want to append, used as we dont want to capture cover page info.
        if loop_count in exclude_page:
            loop_count += 1
        else:
            # reseting the fit of PDFResourceManager for each page and extracting text
            # from the page and appending it to pdf_string.
            resource_manager = PDFResourceManager()
            return_string = StringIO()
            CODENCODING = 'utf-8'
            analysis_parameter = LAParams()
            device = TextConverter(
                resource_manager, return_string, codec=CODENCODING, laparams=analysis_parameter)
            interpreter = PDFPageInterpreter(resource_manager, device)
            interpreter.process_page(page)
            text = return_string.getvalue()
            # check what character is being replaced
            text = re.sub('\\uf0b7', '.', text)
            text = re.sub('\\x0c', '', text)
            text = text.replace("\\", "")
            # remove header and footer
            text = re.search("(?<=CallStreet, LLC).*", text).group(0).strip()
            pdf_string = pdf_string + ' ' + text
            loop_count += 1
    device.close()
    return_string.close()
    # removing disclaimer section
    pdf_string = re.search(
        ".*(?=\sDisclaimer The information)", pdf_string).group(0).strip()
    return pdf_string.strip()


def seperate_sections(section_name, text):
    """
    segment sections of the pdf text into sections as indicated in section_name
    :param section_name: list of strings corresponding where to segment the text
    :param text: the text which we are going to segment
    :return: dictionary with keys corresponding to sections and values corresponding to text in that respective section
    """
    sections = {}
    count = 0
    section_length = len(section_name) - 1
    while count < section_length:
        # use the section stated in section_name as a starting point for where the text is supposed to be extracted from.
        pre_pointer = "(?<=" + section_name[count] + ").*"
        # use the next section stated in section_name as an end point for where the text extracted is supposed to end
        post_pointer = ".*(?=" + section_name[count + 1] + ")"
        text_ = re.search(pre_pointer, text).group(0).strip()
        # text extraction process using the predefined pointers.
        text_ = re.search(post_pointer, text_).group(0).strip()
        # assigning each section its respective text and putting them in sections
        sections[section_name[count]] = text_
        count += 1
    pointer = "(?<=" + section_name[-1] + ").*"
    sections[section_name[-1]] = re.search(pointer, text).group(0).strip()
    return sections


def role(text, exec_titles):
    """
    identify and seperate title from names based on titles in executive_titles.
    :param text: joint text of speaker and title
    :param exec_titles: list of executive titles
    :return: tuple with speaker and its title
    """
    counter = []
    # finding exec titles in the text utilizing regex, and use it to extract name and title.
    for i in exec_titles:
        if text.find(i) > -1:
            counter += [text.find(i)]
    try:
        name_title = (text[:min(counter)].strip(), text[min(counter):].strip())
    except:
        # incase,format changes in future, and not able to extract roles
        name_title = ('', '')
        logging.exception(
            'Cannot extract name and title from {text}'.format(text=text))
    return name_title


# Management discussion section


def management_discussion(section_text, company_name, company_initial, executive_titles):
    """
    produces a tuple with first element being a list of dictionaries with each dictionary in the list corresponding to
    text information regarding to a speaker The second element being a list of dictionaries with each dictionary in the list corresponding to
    speaker information in the management discussion section.
    :param section_text: dictionary with keys corresponding to sections and values corresponding to text in that respective section
    :param company_name: string of company name corresponding to the earnings call pdf tnat is being processed
    :param executive_titles: list of executive titles
    :return: a tuple with first element corresponding to a 2d array with containing management discussion information and second element
    being a dictionary corresponding to the unique participants in this section
    """
    mds_content = []
    text_mds = section_text['MANAGEMENT DISCUSSION SECTION']
    mds_sections = re.split("\.\.\.\.\.\.\.*", text_mds)[1:]
    stripped_mds_sections = [i.strip() for i in mds_sections]
    for paragraph in stripped_mds_sections:
        mds_section_content = []
        second_format = False
        try:
            mds_section_content = mds_section_content + \
                                  [re.search(".*(?=" + company_name + ")", paragraph).group(0).strip()]
            mds_section_content = mds_section_content + \
                                  [re.search("(?<=" + company_name + ").*", paragraph).group(0).strip()]
        except:
            logging.exception(
                'company initial used as company name to divide sections')
            # If company_name has a company_initial instead.
            second_format = True
        if second_format:
            try:
                mds_section_content = mds_section_content + [
                    re.search("^(.*?)" + company_initial, paragraph).group(0).strip()[: -len(company_initial)]]
                mds_section_content = mds_section_content + [
                    re.search("(?<=" + company_initial + ").*", paragraph).group(0).strip()]
            except:
                logging.exception(
                    'unable to seperate speaker title with text in second format')
                continue
        mds_content = mds_content + [mds_section_content]
    # mds_content: [speaker_title, speaker's_content]
    mds_content = np.array(mds_content)
    final_mds_content = []

    for section in mds_content:
        mds_per_paragraph_content = {}
        mds_per_paragraph_content['Speaker'] = role(
            section[0], executive_titles)[0]
        mds_per_paragraph_content['Role'] = role(
            section[0], executive_titles)[1]
        mds_per_paragraph_content['Text'] = section[1]
        final_mds_content += [mds_per_paragraph_content]
        # final_mds_content : List of dictionaries(mds_per_paragraph_content)

    speaker_info = pd.DataFrame(mds_content, columns=['Speaker', 'Text'])
    speaker_info["Role"] = speaker_info.Speaker.apply(
        lambda x: role(x, executive_titles)[1])
    speaker_info["Speaker"] = speaker_info.Speaker.apply(
        lambda x: role(x, executive_titles)[0])
    speaker_info = speaker_info.drop(columns=['Text'])
    speaker_info['Company'] = company_name
    speaker_info = speaker_info.drop_duplicates()
    members_mds = []
    for member in speaker_info.values:
        members = {}
        members['name'] = member[0]
        members['title'] = member[1]
        members['company'] = member[2]
        members_mds += [members]
    return (final_mds_content, members_mds)


# Question and Answer section


def first_occur(text, words):
    """
    helper for qna_text_analysis
    used to capture the first occurence of the letters Q and A.
    :param text: text which at least must contain one word from words
    :param words: words whose positions we want to obtain
    :return: a tuple with an element from words and its corresponding position
    """
    first_occuring = {}
    # iterate through strings that are single capital letters within text
    for match in finditer(" [A-Z] ", text):
        if match.group() in [" " + i + " " for i in words]:
            first_occuring[match.group()] = match.span()[0] + 1
    try:
        type_pos = min(first_occuring.items(), key=lambda x: x[1])
    except:
        # incase,format changes in future, and not able to extract roles
        type_pos = ('', '')
        logging.exception(
            'Cannot extract name and title from {text}'.format(text=text))
    return type_pos


def qna_text_analysis(text_qna, executive_titles):
    """
    rename text_qna with something appropriate
    seperate a question or answer to its type, speaker name, speaker title and the question/answer itself
    :param text_qna: question or answer with speaker credentials
    :param executive_titles: list of executive titles
    :return: a list with type of text(question or answer), name, title, company name and the corresponding Q or A
    """
    # extract the speaker and its tile as they appear before the first comma in text_qna.
    speaker_title = text_qna[:text_qna.find(",")]
    speaker_title = role(speaker_title, executive_titles)
    # finds the index of where the string indicating the type of text exist.
    type_index = first_occur(text_qna, ['Q', 'A'])
    # extracting different type of information through regex
    # strip is used to remove any leading whitespace chracters from the output of first_occur
    type_ = type_index[0].strip()
    # slicing text_qna in a specified format to extract company and text_field
    company = text_qna[text_qna.find(",") + 1: type_index[1]].strip()
    text_field = text_qna[type_index[1] + 1:].strip()
    text_info = [type_, speaker_title[0],
                 speaker_title[1], company, text_field]
    return text_info


def qna_section(qna_raw, executive_titles):
    """
    apply qna_text_analysis on all the text under the QnA section and append the results.
    :param qna_raw: list of text corresponding to questions or answers with speaker credientials in them.
    :param executive_titles: list of executive titles
    :return: a tuple with its first element being 2d array with each element corresponding to the information on each
    text, and second element being a list of dictionaries with information on the askers during in the QnA section.
    """
    qna_content = []
    # iterating through all the qna paragraph block that is in qna_raw
    for qna_text in qna_raw:
        try:
            # extracting information from each qna paragraph block.
            qna_content += [qna_text_analysis(qna_text, executive_titles)]
        except:
            logging.exception(
                'unble to extract qna information from the QnA text block')
            continue
    # converting collected qna infomration to a dataframe
    speaker_info = pd.DataFrame(
        qna_content, columns=['Type', 'Speaker', 'Role', 'Company', 'Text'])
    # Trying to obtain names, companies and titles of those who ask during the qna.
    # filtering on where text block correspond to questions.
    filter_question = speaker_info['Type'] == 'Q'
    speaker_info = speaker_info[filter_question]
    speaker_info = speaker_info.drop(columns=['Type', 'Text'])
    speaker_info = speaker_info.drop_duplicates()
    member_qna = []
    # iterating through rows of unique speakers information and storing information in dictionary.
    for member in speaker_info.values:
        members = {}
        members['name'] = member[0]
        members['title'] = member[1]
        members['company'] = member[2]
        member_qna += [members]
    return qna_content, member_qna


def answer_accumalator(qna_aggregarate):
    """
    helper function for qna_accumalator, map answers to a specific question.
    :param qna_aggregarate: take entire array from qna_section where each element is a tuple with first element
    being either "Q" or "A" and second element is the content of the QnA
    :return: return position of where last type is 'A' text relative to the previous occurence of type 'Q' text
    """
    # list of 'Q' and 'A' where first element must always be a 'Q' due to how it is called in parent function.
    qna_count = []
    for element in qna_aggregarate:
        qna_count += element[0]
    # keep track how many 'Q' and 'A's have been encountered
    q_count = 0
    a_count = 0
    for qna in qna_count:
        if q_count == 2:
            # stops when finding out the second Q as it determines where to slice for each QnA block
            break
        elif qna == 'Q':
            q_count += 1
        elif qna == 'A':
            a_count += 1
    slicing_index = a_count + 1
    # index at which a QnA block appars.
    return slicing_index


def qna_accumalator(qna):
    """
    produces a list of n dictionaries for all n questions in QnA section with elements in key,value pairs in each
    dictionary holding answer information for that specific question.
    :param qna: 2d array with each element corresponding to the information on each text, where each element is a tuple with first element
    being either "Q" or "A" and second element is the content of the QnA
    :return: list of dictionaries holding information regarding to the questions
    """
    questions_info = []
    # function is placed under try catch block on parent function in case no answer is tied to a question.
    # form qna blocks within qna section.
    while len(qna) > 0:
        # checks if second element is a question type of text and slice qna to start from second element if true
        if qna[1][0] == 'Q':
            qna = qna[1:]
        # checks if last element is question type and excludes it if it is.
        elif qna[-1][0] == 'Q':
            qna = qna[:-1]
        # if first element is question type text, we want to slice qna to capture answers that correspond to a certain question
        elif qna[0][0] == 'Q':
            slicing_index = answer_accumalator(qna)
            ###
            # captures one qna block based on slicing index
            one_qna = qna[:slicing_index]
            # captures all question info for that qna block
            question_info = {}
            question_info['Q_speaker'] = one_qna[0][1]
            question_info['Q_title'] = one_qna[0][2]
            question_info['Q_firm'] = one_qna[0][3]
            question_info['Question_text'] = one_qna[0][4]
            question_info['Answer'] = []
            # one qna block may have multiple answers, hence capturing all answer information for that qna block.
            for answer in one_qna[1:]:
                answer_info = {}
                answer_info["A_speaker"] = answer[1]
                answer_info["A_title"] = answer[2]
                answer_info["A_text"] = answer[4]
                question_info['Answer'] += [answer_info]
            questions_info += [question_info]
            ###
            # reducing qna by slicing at slicing index to stop while loop
            qna = qna[slicing_index:]
    return questions_info


def final_remarks(qna_texts, company_name, executive_titles):
    """
    extract information from the final_remarks section
    :param qna_texts: list of text from QnA section
    :param company_name: string of company name corresponding to the earnings call pdf tnat is being processed
    :param executive_titles: list of executive titles
    :return: a dictionary containing infomration last remarks.
    """
    # multiple formats may exist
    second_format = False
    try:
        # assumes final_remarks is the second last text block within qna section
        final_remarks_text = qna_texts[-2]
        # capturing all infomration regarding final section.
        speaker_title = re.search(
            ".*(?=" + company_name + ")", final_remarks_text).group(0).strip()
        text = re.search("(?<=" + company_name + ").*",
                         final_remarks_text).group(0).strip()
        speaker_title = role(speaker_title, executive_titles)
        final_remarks = {
            'Speaker': speaker_title[0], 'Title': speaker_title[1], 'Text': text}
    except:
        second_format = True
    if second_format:
        try:
            # assumes final_remarks is the last text block within qna section
            final_remarks_text = qna_texts[-1]
            # capturing all infomration regarding final section.
            speaker_title = re.search(
                ".*(?=" + company_name + ")", final_remarks_text).group(0).strip()
            text = re.search("(?<=" + company_name + ").*",
                             final_remarks_text).group(0).strip()
            speaker_title = role(speaker_title, executive_titles)
            final_remarks = {
                'Speaker': speaker_title[0], 'Title': speaker_title[1], 'Text': text}
        except:
            logging.exception('unble to extract final_remarks')
    return final_remarks


def pdf_parse(earnings_call_file):
    """
    acts as a wrapper of all the functions
    :param earnings_call_file: File - representation of the earnings call file
    :return: json object with all relevant information extracted from the pdf.
    """
    # exec_titles are needed to parse speaker info
    exec_titles = ['Head', 'President', 'Vice', 'Chairman', 'Chief', 'Senior', 'Analyst', 'Group', 'Financial',
                   'Executive', 'U.S.']
    final_json = {}
    earnings_call_file = open(earnings_call_file, "rb")

    ### Cover Page Extraction ###
    # extracts all information from the coverpage and use parse info to assign basic information
    # set the seek to 0 to have the processing done from beginning of the file
    #earnings_call_file.stream.seek(0)
    COVER_PAGE = coverpageinfo(earnings_call_file)
    final_json['company'] = {
        "name": COVER_PAGE["company_name"], "symbol": COVER_PAGE["symbol"]}
    final_json['quarter'] = COVER_PAGE['quarter_year'][:2]
    final_json['year'] = int(COVER_PAGE['quarter_year'][-4:])
    final_json['call_date'] = COVER_PAGE['published_date']

    # Append PDF and seperate sections
    # exclude first page and list of speakers page(we already capture all this info)
    # set the seek to 0 to have the processing done from beginning of the file
    #earnings_call_file.stream.seek(0)
    pdf_appended = append_pdf_pages(earnings_call_file, [0])
    pdf_text_content = seperate_sections(['MANAGEMENT DISCUSSION SECTION', 'QUESTION AND ANSWER SECTION'],
                                         text=pdf_appended)
    mgmt_discuss_section = management_discussion(pdf_text_content, final_json['company']["name"],
                                                 final_json['company']["symbol"], executive_titles=exec_titles)
    final_json["management_discussion_section"] = mgmt_discuss_section[0]

    # QnA Section
    # qna is a single string and we use regex to split up to questions and answers
    qna_raw = [qna.strip() for qna in re.split(
        "\.\.\.\.\.\.\.*", pdf_text_content["QUESTION AND ANSWER SECTION"])]
    qnas = qna_section(qna_raw, executive_titles=exec_titles)
    qna_final = qna_accumalator(qnas[0])
    final_json["question_and_answers"] = qna_final

    # Participants list
    final_json["participants"] = mgmt_discuss_section[1]
    final_json["other_participants"] = qnas[1]
    format_second = False

    # Checking for Final_remarks
    try:
        final_remarks_ = final_remarks(
            qna_raw, final_json['company']["name"], executive_titles=exec_titles)
        final_json['final_remarks'] = final_remarks_
    except:
        logging.exception('no final_remarks section found')
        pass

    return final_json

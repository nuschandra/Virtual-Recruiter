from pdf_reader import create_data
import spacy
import warnings
from pymongo import MongoClient
import uuid
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import os

nlp=spacy.load('resume_model')

def convert_pdf_to_txt(path):
    resource_manager = PDFResourceManager()
    device = None
    try:
        with StringIO() as string_writer, open(path, 'rb') as pdf_file:
            device = TextConverter(resource_manager, string_writer, codec='utf-8', laparams=LAParams(line_margin=1))
            interpreter = PDFPageInterpreter(resource_manager, device)

            for page in PDFPage.get_pages(pdf_file):
                interpreter.process_page(page)

            pdf_text = string_writer.getvalue()
    finally:
        if device:
            device.close()
    print(pdf_text)
    return pdf_text

directory = os.getcwd()
resume_dir = os.path.join(directory,"TestResumes")
c=0
for files in os.listdir(resume_dir):
    file_path=os.path.join(resume_dir,files)
    text=convert_pdf_to_txt(file_path)
    f=open("TestResumeResults/resume"+str(c)+".txt","w")
    doc_to_test=nlp(text)
    resume_dict={}
    for ent in doc_to_test.ents:
        resume_dict[ent.label_]=[]
    for ent in doc_to_test.ents:
        resume_dict[ent.label_].append(ent.text)

    #save_candidate(resume_dict)
    for i in set(resume_dict.keys()):

        f.write("\n\n")
        f.write(i +":"+"\n")
        for j in set(resume_dict[i]):
            f.write(j.replace('\n','')+"\n")
        
    c+=1
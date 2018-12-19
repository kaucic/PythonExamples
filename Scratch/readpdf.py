# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 19:16:02 2016

@author: Kimberly
"""

import sys
import re

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure

def find_list_pattern(lst,pattern,nlines):
    """Search the list for a particular pattern and then return the next nlines elements in the list"""
    n = len(lst)
    for i in range(n):
        found = re.search(pattern,lst[i])
        if found:
            print "Found", pattern, "in textbox=", lst[i]
            last = min(n,i+nlines)
            return lst[i:last]

def find_pattern(txt,pattern,nchars):
    before, found, after = txt.partition(pattern)
    n = min(len(after),nchars)
    return after[:n]

def parse_layout(layout):
    """Function to recursively parse the layout tree."""
    # Create string of all the text
    txt = ""
    for lt_obj in layout:
        #print(lt_obj.__class__.__name__)
        #print(lt_obj.bbox)
        if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
            #print lt_obj.get_text()
            val = lt_obj.get_text()
            val.replace(u'\xa0', ' ').encode('utf-8')
            txt += val
        elif isinstance(lt_obj, LTFigure):
            txt += parse_layout(lt_obj)  # Recursive
            
    return txt
            
def main(fname):
    with open(fname,'rb') as fd:
        parser = PDFParser(fd)
        doc = PDFDocument(parser)

        # Check if document is extractable, if not abort
        if not doc.is_extractable:
            raise Exception

        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        all_txt = ""
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            layout = device.get_result()
            txt = parse_layout(layout)
            all_txt += txt
            
        #print "Converted text\n", all_txt
        snip = find_pattern(all_txt,"volunteer recycling",200)
        print snip
        
if __name__ == '__main__':
    main(sys.argv[1])
    
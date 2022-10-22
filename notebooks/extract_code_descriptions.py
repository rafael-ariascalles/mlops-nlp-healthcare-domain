# imports 
import pandas as pd
from bs4 import BeautifulSoup
import re
import glob
import json
import time
import warnings
warnings.simplefilter("ignore")

#load data files
files = glob.glob("data/bs4_l4_dump/*")

# initate empty json file
icd_json = []

# initaite counters
num_blank_clinical_info = 0
num_blank_synonyms = 0
counter = 0

# iterate through files and start timer 
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))
for file in files:

    data = pd.read_pickle(file)
    soup = BeautifulSoup(data)

    pattern = re.compile(r'should not be used for reimbursement purposes')
    pattern2 = re.compile(r'code that can be used to indicate a diagnosis for reimbursement purposes')

    icd_code = file.split('_')[-1].replace('.pkl', '')
    
    try:
        # if code is able to be reimbursed 
        if soup.find(text=pattern) == None and soup.find(text=pattern2) != None:
        
            # extract code description
            code_description = soup.findAll('h2', {'class': 'codeDescription'})[0].text
            
            # extract clinical information if avaliable 
            try:
                clinical_information = soup.find(text='Clinical Information').find_all_next()[0].text
            except:
                clinical_information = None
                num_blank_clinical_info += 1
            
            # extract approximate synonums if avaliable 
            try:
                approximate_synonyms = soup.find(text='Approximate Synonyms').find_all_next()[0].text
            except:
                approximate_synonyms = None
                num_blank_synonyms += 1
            
            # add code data to json 
            icd_json.append({'icd10Code': icd_code,
                             'text': {'codeDescription':code_description,
                                                     'clinicalInformation':clinical_information,
                                                     'approximateSynonyms':approximate_synonyms}
                             })
            
            counter += 1
    
            if counter % 1000 == 0:
                print(f'Completed loading {counter} codes')
                print("--- %s seconds ---" % (time.time() - start_time))

    except:
        print(f'failed on {icd_code} code')
        break


# print stats 
print(f'\nCompleted loading {len(icd_json)} icd codes') 
print(f'{round((len(icd_json)/ len(files)) * 100, 2)} % of codes captured')
print(f'{round((num_blank_clinical_info/len(icd_json)) * 100, 2)} % of codes had no clinical information')
print(f'{round((num_blank_synonyms/len(icd_json)) * 100, 2)} % of codes had no approximate synonyms')

# save json 
with open("data/icd_json.json", "w") as outfile:
    json.dump(icd_json, outfile)


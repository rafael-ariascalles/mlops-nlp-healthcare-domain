import json

def get_disease_mapping(path='mapping/disease_icd_mapping.json'):

    json_data = []
    with open(path, 'r') as handle:
        for line in handle:
            line = line.strip("\n").strip('{').strip('}').strip(',').replace('"', "")
            if len(line) > 0:
                key = line.split(':')[0]
                value = line.split(':')[1].strip('[').strip(']').split(',')
                json_data.append({key:value})
                
    return(json_data)


def get_codes_per_dx(dx, disease_map):
    
    icd9_codes = []
    
    for d in disease_map:
        if dx in d:
            icd9_codes.append(list(d.values())[0][0])
            
    return(icd9_codes)


def get_chapter_map(path='mapping/chapter_description.json'):
    
    # Opening JSON file
    f = open(path)
    data = json.load(f)
    f.close()
    return(data)


def prediction_json(prediction):
    
    prediction = prediction['children'][1]['children']
    prediction.sort(key=lambda x: x['value'], reverse=True)
    
    chapter_map = get_chapter_map()
    disease_map = get_disease_mapping()
    
    output = {}
    for i, p in enumerate(prediction):
        p['value'] = str(p['value'])
        p['description'] = chapter_map[p['name']]
        p['children'].sort(key=lambda x: x['value'], reverse=True)
        for p2 in p['children']:
            p2['value'] = str(p2['value'])
            p2['codes'] = get_codes_per_dx(p2['name'], disease_map)
            
        output['Prediction Ranking: ' + str(i+1)] = p
        
    return(output)
     
     
     
     
# def disease_output(d):
#     s = f"""
#     ICD9 DX: {d['name']} 
#     Probability Score: {round(float(d['value']), 2)}
#     Associated ICD9 Codes: {get_codes_per_dx(d['name'])}"""
#     return(s)
     
     
#     chapter_map = get_chapter_map()
#     output = ''
#     for i in range(0, 4):
    
#         output += f"""
# Chapter Ranking: {i+1}
# ICD9 DX chapter: {preds[i]['name'].replace('_','-')} 
# Description: {chapter_map[preds[i]['name']]}
# Probability score: {round(float(preds[i]['value']), 2)} 
# """

#         output += f"""
#     Top diseases associated with ICD9 DX chapter {preds[i]['name'].replace('_','-')}:"""
#         for r in preds[i]['children']:
#             output+='\n'
#             output+=disease_output(r)
#         output+='\n\n'
    

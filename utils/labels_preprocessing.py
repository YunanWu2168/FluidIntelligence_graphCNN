import pandas as pd
import numpy as np
import os

import scipy.io as sio

from datetime import datetime

def load_raw_labels(path):
    labels = sio.loadmat(path)
    labels = pd.DataFrame(labels['diag'], columns = ['RID', 'EXAMDATE', 'DX'])
    
    def remove_paren(value):
        ''' Helper function for removing parentheses in all of the dataframe's cells
            
            Args:
                value (nested list): expected to loop over and call the "first" element until a single number is returned
            Returns:
                value (single element)
        '''
        i=0
        while i<num:
            value=value[0]
            i+=1
        return value
    
    num=2                                           # the values in the RID column are nested within 2 parentheses
    labels.RID = labels.RID.apply(remove_paren)
    labels = labels.astype({'RID': 'int32'})
    
    num=1                                           # the values elsewhere are nested each within one set of parentheses
    labels.EXAMDATE = labels.EXAMDATE.apply(remove_paren)
    labels.DX = labels.DX.apply(remove_paren)
    
    labels = labels.astype({'RID':'int32', 'EXAMDATE':str, 'DX':str})
    def change_dt_format(dt_value):
        ''' Read the data string and convert it to a datetime object
            
            Args: 
                dt_value (str): expects string of scan EXAMDATE and converts it into a datetime object
            
            Returns:
                datetime object interpretation of the read-in date
        '''
        if pd.isnull(dt_value):
            return np.nan
        return datetime.strptime(dt_value, "%Y-%m-%d")
    
    labels.EXAMDATE = labels.EXAMDATE.apply(change_dt_format)
    return labels

def assert_single_scan_per_visit(data_dir):
    ''' Function for getting the path to each respective patient in our database, assuring that one scan per visit
        
        Args: data_dir (str): path to the directory of the data we're interested in 
    '''
    assert os.path.isdir(data_dir)
    filenames = [f for f in os.listdir(data_dir) if f[:4] == 'ADNI'] # get MRI filenames
    
    scans = []
    for idx,file in enumerate(filenames):
        file = file.split('_')
        assert file[0] == 'ADNI' # filenames all start with 'ADNI'
        
        assert file[1].isdigit() # followed by their PTID (last 4 digits are their RID)
        assert file[2].isalpha()
        RID = file[3]
        
        # assume the date comes right after (search for the largest string, its first 8 characters are the date)
        date = file[4]
        for part in file[5:]:
            if len(part) < 2: # don't even consider strings less than 2 characters for dates, S#####, I#######
                continue
            
            if part[0] == 'S' and part[1:].isdigit():   # visit ID
                visit = part[1:]
            elif part[0] == 'I' and part[1:].isdigit(): # instance ID from that visit (once this is obtained, break out)
                instance = part[1:]
                break
            elif len(part) > len(date) and part.isdigit(): # the date
                date = part
        
        date = datetime.strptime(date[:8], "%Y%m%d")
        # append row of scans dataframe
        scans.append({"RID": int(RID), 
                      "FILEDATE": date,
                      "Visit": int(visit),
                      "Instance": int(instance),
                      "Path": filenames[idx]})
    scans = pd.DataFrame(scans)
    
    # determine sessions that had multiple scans on the same day (keep the last)
    dups = scans[scans.duplicated(subset=['RID', 'FILEDATE'], keep=False)]
    dups = dups.sort_values(by = ['RID', 'FILEDATE', 'Visit', 'Instance'], ascending = False)
    dups = dups.reset_index(drop=True)
    dups = dups[dups.duplicated(subset = ['RID', 'FILEDATE'], keep = 'first')]
    
    # determine sessions that only have a single scan
    unique = scans.drop_duplicates(subset=['RID', 'FILEDATE'], keep=False)
    
    dups_paths = dups.Path.to_list()
    for u in unique.Path.to_list():
        assert u not in dups_paths
    return pd.concat([dups, unique], ignore_index = True).drop(columns=['Visit', 'Instance'])

def cut_down_labels(avail, overall):
    ''' Function for cutting down the number of overall labels to the number of available scans
        
        Args:
            - avail (pd.DataFrame) = dataframe of available scans and their paths
            - overall (pd.DataFrame) = dataframe of available labels from ADNI (provided by Arjun Punjabi)
    '''
    
    def date_differ(date1, date2):
        '''Helper function for computing the number of days between two dates using datetime objects'''
        delta = date1 - date2
        return abs(delta.days)
    
    cut_down = []
    for idx_avail, scan in avail.iterrows():
        sub = overall[overall.RID == scan.RID] # get sub-dataframe
        
        '''go to the next available scan if there's no match in the labels for it'''
        if sub.empty:
            continue
        
        potent_matches = sub.EXAMDATE.to_list() # list of potential matches
        
        # compute the numebr of days between each potential match and file date and take the smallest difference
        delta = [date_differ(scan.FILEDATE, date) for date in potent_matches]
        delta_idx = np.argmin(delta)
        
        if delta[delta_idx] <= 60:
            DX = sub.DX.to_list()[delta_idx]                  # extract diagnosis for that particular match
            cut_down.append({'FILEPATH':scan.Path,            # add row to new dataframe
                             'RID':scan.RID,
                             'EXAMDATE':scan.FILEDATE,
                             'DX':DX})
    
#     return pd.DataFrame(cut_down).astype({'EXAMDATE':str})
    return pd.DataFrame(cut_down)

def load_labels(labels_path, data_dir):
    labels = load_raw_labels(labels_path)
    avail = assert_single_scan_per_visit(data_dir)
    return cut_down_labels(avail, labels)
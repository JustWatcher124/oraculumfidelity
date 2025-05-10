import pandas as pd



def get_list_license_plates(path):
    df = pd.read_csv(path)
    return df['krz'].to_list()


    

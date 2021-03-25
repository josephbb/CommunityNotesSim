import sqlalchemy
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
from src.segmentation import aggregate
import os
tqdm.pandas()
pandarallel.initialize(nb_workers=8,verbose=True)

def get_engine(venus_path='/home/joebak/venus_cred.txt'):
    with open(venus_path) as f:
        venus_auth = f.readlines()
    venus_auth = [x.strip() for x in venus_auth]

    connection_string = 'postgresql://{user}:{password}@{host}/{database}'
    engine = sqlalchemy.create_engine(connection_string.format(user=venus_auth[1],
                                                               password=venus_auth[3],                                                                                  host=venus_auth[0],
                                                               database=venus_auth[2]))
    return engine


def list_incidents(engine):
    """Return a list of all incidents in our database.
    Keyword arguments:
    engine -- postgres engine created with src.database.get_engine
    """
    query_all = "SELECT DISTINCT incident FROM public.all_ticket_tweets WHERE incident IS NOT NULL;"
    incidents =pd.read_sql(query_all, con=engine)
    return incidents


def get_incident_data(incident,engine):
    """Return user_followers_count, user_screen_name, created_at, and user_verified 
     for and incident.
    Keyword arguments:
    incident -- the name of an incident, as identified in our database
    engine -- postgres engine created with src.database.get_engine
    """
    query = "SELECT  user_followers_count, user_screen_name, created_at, user_verified FROM all_ticket_tweets WHERE incident=(%(incident)s)"
    incident_df = pd.read_sql(query, params={'incident':incident},con=engine)
    return incident_df


def aggregate_and_save(row, engine,freq=5, removed=[], floc='/data/timeseries/aggregated/', root='.',keep=True):
    """Gather an incident's raw data, aggregates it (with removed users) and save it.
    Keyword arguments:
    row -- a row of a pandas table created by list_incidents
    engine -- postgres engine created with src.database.get_engine
    removed -- a list of removed users by user_screen_name 
    floc -- relative path to store output
    root -- root of path to store output. Default '.'
    """
    if keep and os.path.isfile(root + floc + row['incident_name'] + '_raw.parquet'):
        pass
        
    else:
        raw_df = get_incident_data(row['incident'],engine)
        if type(removed)==dict:
            removed = removed[row['incident']]   
        agg = aggregate(raw_df,freq, removed)
        agg.to_parquet(root + floc + row['incident_name'] + '_raw.parquet',compression=None)
    return True
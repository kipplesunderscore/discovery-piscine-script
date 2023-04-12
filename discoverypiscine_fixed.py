#!/usr/bin/env python3

from datetime import date
import pandas as pd
import numpy as np
import re

from dotenv import load_dotenv
load_dotenv()

import os

# making API requests
import json
import requests
import requests_cache

# python decorators to make easy rate limited functions
from ratelimit import limits, RateLimitException, sleep_and_retry

import math
from datetime import date
import pandas as pd
import numpy as np

# list of user logins to exclude from the data
EXCLUDE_USERS = [
    "jbrown", 
    "codeathon01", 
    "codeathon02", 
    "codeathon03", 
    "codeathon04", 
    "42adel-event-01",
    "discovery06test"
    ]

# should be set so that we can filter down the output of various API requests
# so they don't take forever
START_DATE = '2023-04-04'

# normally you should use this however if you are doing historic data / testing
# you will have to set manually

# END_DATE = str(date.today())
END_DATE = '2023-04-15'

# this is used for the ratelimiting by default applications can make
# 2 calls per second, if you have increased limits modify these values.
# there is also a limit of 1200 / hour if you see errors it could be this.
# set to 1 call per second to minimise rate limit chances.
MAX_CALLS_PER_PERIOD=1
PERIOD_SECONDS=1

CAMPUS_ID = 36 # adelaide
CURSUS_ID = 3 # discovery piscine

# 1 hour
CACHE_EXPIRY=3600

requests_cache.install_cache(cache_name='42_api_cache',  backend='sqlite', expire_after=CACHE_EXPIRY)

uid = os.getenv("42_CLIENT_UID")

secret = os.getenv("42_CLIENT_SECRET")

api_url = 'https://api.intra.42.fr'

def get_access_token():
    token_url = '{}/oauth/token'.format(api_url)
    response = requests.post(token_url, 
                             data=
                             {'grant_type': 'client_credentials', 
                              'client_id': uid, 
                              'client_secret': secret})
    json_response = json.loads(response.text)
    return json_response

access_token = get_access_token()['access_token']

headers = {'Authorization': 'Bearer {}'.format(access_token)}

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_PERIOD,period=PERIOD_SECONDS)
def get_endpoint(endpoint, data = {}, params={}):
    print(f'making request to {endpoint}')
    endpoint_url = '{}{}'.format(api_url, endpoint)
    page = 1
    if 'page' in params:
        page = int(params['page'])
    if 'page[number]' in params:
        page = int(params['page[number]'])
    response = requests.get(endpoint_url, 
                            data=data, 
                            headers=headers, 
                            params=params)
    total_pages = 1
    if 'X-Total' in response.headers:
        total_count = int(response.headers['X-Total'])
        per_page = int(response.headers['X-Per-Page'])
        total_pages = int(math.ceil(total_count / per_page))
    json_response = json.loads(response.text)
    if page < total_pages:
      if 'page[number]' in params:
        params['page[number]'] = page + 1
      else:
        params['page'] = page + 1
      return json_response + get_endpoint(endpoint, data, params)
    return json_response

# get all users participating in the cursus
dp_users = get_endpoint(f'/v2/cursus/{CURSUS_ID}/cursus_users', params={'filter[campus_id]': CAMPUS_ID, 'filter[active]': 'true', 'per_page': 100})

# additional filtering to try and remove un wanted users
dp_users = [u for u in dp_users if u['user']['staff?'] == False and u['user']['pool_month'] != None and u['user']['kind'] == 'student' and u['user']['login'] not in EXCLUDE_USERS]

def clean_scale_event(user, se):
    return {'user_id': user['id'],
            'login': user['login'],
           'project_id': se['team']['project_id'],
            'status': se['team']['status'],
            'final_mark': se['team']['final_mark'],
            'closed_at': se['team']['closed_at'],
            'filled_at': se['filled_at']
           }

def get_scale_event(user):
    user = user['user']
    scale_events = get_endpoint(
        f'/v2/users/{user["id"]}/scale_teams/as_corrected',
        params = 
        {'filter[cursus_id]': CURSUS_ID,
         'per_page': 100
         })
    cleaned = []
    for se in scale_events:
        cleaned.append(clean_scale_event(user, se))
    return cleaned
    
def flatten(l):
    return [item for sublist in l for item in sublist]

all_scale_events = flatten(list(map(get_scale_event, dp_users)))

# convert all_scale_events to a pandas DataFrame
scale_events_df = pd.DataFrame.from_records(all_scale_events)

# TODO: look into making this faster
# might be able to lookup all projects for cursus
# and use that as a map instead

project_id_map = {}

def get_project_name(id):
    if id in project_id_map:
        return project_id_map[id]
    project = get_endpoint(f'/v2/projects/{id}')
    project_id_map[id] = project['name']
    return project['name']

# User Level
# retrieves the user's level from the campus_users based on their user id.

def get_user_level(user_id):
    return next(item for item in dp_users if item['user']['id'] == user_id)['level']

def missing_users(df):
    for user in dp_users:
        if not user['user']['login'] in df.login.values:
            print(user['user']['login'])

# Scale Event Tweaks
# here we take the scale event DataFrame and tweak it so that the column types make sense so we can do the cool manipulations / analysis with less hassle.

def extract_project_level_from_name(name):
  return re.sub('[a-zA-Z\s\-]', '', name)

def tweak_scale_events(df):
    return (df
            # fill blank final_mark with 0 and convert to int
            .assign(final_mark=lambda df_:(df_.final_mark.fillna(0).astype('uint8')))
            # count the number of tries the user has had
            .assign(evaluations=lambda df_:(df_.groupby('user_id')['user_id'].transform('count').astype('uint8')))
            # create new column for Project Name based on project_id
            .assign(project_name=lambda df_:(df_.apply(lambda x: get_project_name(x.project_id), axis=1)))
            # get the user level out of the campus_users data (see above function)
            .assign(level=lambda df_:(df_.apply(lambda x: get_user_level(x.user_id), axis=1)))
            .assign(project_level=lambda df_:(df_.apply(lambda x: extract_project_level_from_name(get_project_name(x.project_id)), axis=1)))
            # convert datetime data from strings to real datetime types
            .assign(closed_at=lambda df_:(pd.to_datetime(df_.closed_at).dt.tz_convert('Australia/Adelaide')))
            .assign(filled_at=lambda df_:(pd.to_datetime(df_.filled_at).dt.tz_convert('Australia/Adelaide')))
            # status & project_name are categorical (think enums)
            .astype({'status': 'category',
                     'project_name': 'category',
                     'project_level': 'float64'
                     })
            # don't really need these anymore so drop them
            .drop(columns=['user_id', 'project_id'])
           )

# current final mark per project
tweak_scale_events(scale_events_df).pivot_table(index=['login'], columns='project_name', values='final_mark', aggfunc='max', fill_value=0).to_csv(f'grade_per_project-{date.today()}.csv')

# final mark per project per day
tweak_scale_events(scale_events_df).pivot_table(index = ['login', 'project_name'], columns=pd.Grouper(freq='D', key='filled_at'), values='final_mark').fillna(method='pad', axis=1).fillna(0).to_csv(f'grade_per_project_per_day-{date.today()}.csv')

# evaluations per project per day
tweak_scale_events(scale_events_df).pivot_table(index = ['login', 'project_name'], columns=pd.Grouper(freq='D', key='filled_at'), values='project_level', aggfunc='count').to_csv(f'evaluations_per_project_per_day-{date.today()}.csv')

# farthest project per day
tweak_scale_events(scale_events_df).pivot_table(index=['login'], columns=pd.Grouper(freq='D', key='filled_at'), values='project_level', aggfunc='max').fillna(method='pad', axis=1).fillna(0).to_csv(f'project_progression_per_day-{date.today()}.csv')

# user level
tweak_scale_events(scale_events_df).drop_duplicates(subset=['login']).set_index('login')[['level']].to_csv(f'user_level-{date.today()}.csv')

# this code is batched to avoid long URLs which seem to cause the API
# to throw 502 errors, code will do 30 user_ids at a time which is
# conservative

users_ids = [str(i['user']['id']) for i in dp_users]

locations = []

BATCH_SIZE = 30

for i in range(0, len(users_ids), BATCH_SIZE):
  user_id_filter = ','.join(users_ids[i:i + BATCH_SIZE])
  locations.extend(get_endpoint(f'/v2/campus/{CAMPUS_ID}/locations', 
                         params={
                             'filter[user_id]' : user_id_filter,
                             'page[size]' : 100,
                             'page[number]' : 1,
                             'range[begin_at]': f'{START_DATE},{END_DATE}',
                          }))

# print(f'number of locations {len(locations)} ({START_DATE,END_DATE})')

def clean_location(l):
    return { 
        'user_id': l['user']['id'],
        'login': l['user']['login'],
        'begin_at': l['begin_at'],
        'end_at': l['end_at']
    }

cleaned_locations = list(map(clean_location, locations))

locations_df = pd.DataFrame.from_records(cleaned_locations)

def missing_date_range(df_):
    return pd.concat([df_, (
            pd.date_range(min(df_.columns), max(df_.columns), inclusive='left')
            .to_frame()
            .reset_index(drop=True)
            .rename(columns={0:'login'})
            .set_index('login').T
        )]).pipe(lambda d: d.reindex(sorted(d.columns), axis=1))

def tweak_locations(df):
    return (df
            .assign(begin_at=lambda df_:(pd.to_datetime(df_.begin_at).dt.tz_convert('Australia/Adelaide')))
            .assign(end_at=lambda df_:(pd.to_datetime(df_.end_at).dt.tz_convert('Australia/Adelaide')))
            .assign(delta=lambda df_:(df_.end_at - df_.begin_at))
            #.pipe(lambda df_: df_.groupby(['login', pd.Grouper(freq='D', key='begin_at')])['delta'].sum().to_frame().reset_index())           
            #.pivot_table(index=['login'], columns='begin_at', values='delta', aggfunc='sum')
            #.rename_axis(None,axis=1)
            #.reset_index()
            #.pipe(missing_date_range)
            #.fillna(pd.Timedelta(0))
           )

#a = tweak_locations(ldf)
#idx = tweak_locations(ldf).columns[1:].astype('datetime64[ns]')
#p = pd.period_range(min(idx), max(idx))
#idx.reindex(p)[0]

# tweak_locations(locations_df)

# minutes per day
(tweak_locations(locations_df)
            .pipe(lambda df_: df_.groupby(['login', pd.Grouper(freq='D', key='begin_at')])['delta'].sum().to_frame().reset_index())           
            .pivot_table(index=['login'], columns='begin_at', values='delta', aggfunc='sum')
            .rename_axis(None,axis=1)
            .reset_index()
            .set_index('login')
            .pipe(missing_date_range)
            .fillna(pd.Timedelta(0))
            .apply(lambda x: x.dt.total_seconds() / 60)
            .select_dtypes('float64').astype('int64')
).to_csv(f'minutes_per_day-{date.today()}.csv')

# minutes per week
(tweak_locations(locations_df)
            .pipe(lambda df_: df_.groupby(['login', pd.Grouper(freq='W', key='begin_at')])['delta'].sum().to_frame().reset_index())           
            .pivot_table(index=['login'], columns='begin_at', values='delta', aggfunc='sum')
            .rename_axis(None,axis=1)
            .reset_index()
            .set_index('login')
            #.pipe(missing_date_range)
            .fillna(pd.Timedelta(0))
            .apply(lambda x: x.dt.total_seconds() / 60)
            .select_dtypes('float64').astype('int64')
).to_csv(f'minutes_per_week-{date.today()}.csv')

# attendance
(tweak_locations(locations_df)
            .pipe(lambda df_: df_.groupby(['login', pd.Grouper(freq='D', key='begin_at')])['delta'].sum().to_frame().reset_index())           
            .pivot_table(index=['login'], columns='begin_at', values='delta', aggfunc='sum')
            .rename_axis(None,axis=1)
            .reset_index()
            .set_index('login')
            .pipe(missing_date_range)
            .notnull()
).to_csv(f'attendance_per_day-{date.today()}.csv')

print('done')

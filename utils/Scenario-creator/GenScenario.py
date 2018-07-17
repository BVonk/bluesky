# -*- coding: utf-8 -*-

import sys
sys.path.append('../../bluesky/tools/')
from geo import qdrdist
from aero import vtas2cas
from so6_to_scn import readFile
from DatToScn import DatToScn


import datetime
import pandas as pd
import numpy as np

def find_intersect(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    """
    Find intersection point between two vectors
    """
    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y
    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / ((-s2_x * s1_y + s1_x * s2_y))
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / ((-s2_x * s1_y + s1_x * s2_y))
    if (s >= 0 and s <= 1 and t >= 0 and t <= 1):
        # Collision detected
        x = p0_x + (t * s1_x+0.00001)
        y = p0_y + (t * s1_y+0.00001)
        return 1, x, y

    return 0, None, None # No collision

def format_time(td):
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{}:{}:{}.00>'.format(str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2))

def convert_lat(lat):
    sign = 1 if lat[0]=='N' else -1
    lat=lat[1:].split('.')
    lat = float(lat[0]) + float(lat[1])/60 + float(lat[2])/60
    return sign*lat

def convert_lon(lon):
    sign = 1 if lon[0]=='E' else -1
    lon=lon[1:].split('.')
    lon = float(lon[0]) + float(lon[1])/60 + float(lon[2])/60
    return sign*lon



names = ('origin', 'destination', 'actype', 't1', 't2', 'fl1',
         'fl2', 'status', 'callsign', 'date1', 'date2',
         'lat1', 'lon1', 'lat2', 'lon2', 'flightid',
         'sequence', 'length')
usecols = np.arange(1,19)

dtype = {'origin': str,
         'destination': str,
         'actype': str,
         't1': str,
         't2': str,
         'fl1': int,
         'fl2': int,
         'status': int,
         'callsign': str,
         'date1': str,
         'date2': str,
         'lat1': np.float32,
         'lon1': np.float32,
         'lat2': np.float32,
         'lon2': np.float32,
         'flightid': int,
         'sequence': int,
         'length': np.float32}


fn = '20180705_20180705_0000_2359__EHAM___m3'
df = pd.read_csv('tempData/' + fn + '.so6', sep=' ', names=names, usecols=usecols, dtype=dtype)#.iloc[0:500,:]


print('Converting timestamps...')
df.t1 = (df.date1 + df.t1).apply(lambda x: '20'+x)
df.t2 = (df.date2 + df.t2).apply(lambda x: '20'+x)
#df.drop()
#pd.DataFrame.drop_duplicates()
df.t1 = pd.to_datetime(df.t1, infer_datetime_format=True)
df.t2 = pd.to_datetime(df.t2, infer_datetime_format=True)

#Prepare variables
df.lat1 /= 60
df.lon1 /= 60
df.lat2 /= 60
df.lon2 /= 60
df.fl1 *= 100
df.fl2 *= 100
df['hdg'], df['dist'] = qdrdist(df.lat1, df.lon1, df.lat2, df.lon2)
t_segment = df.t2-df.t1
t_segment = t_segment.apply(lambda x: float(abs(x.seconds)+0.001))
df['vs'] = 60.0 / t_segment * (df.fl2 - df.fl1)
df['gs'] = 3600.0 / t_segment * df.dist
df['cas'] = vtas2cas(df.gs, df.fl1*0.3048)



print('Filtering FIR...')
flightids = df.flightid.unique()
aclist=[]

## Filter FIR Data
fir = pd.read_csv('../../data/navdata/fir/EHAA.txt', delimiter=' ', names=['lat', 'lon'])
fir.drop_duplicates(inplace=True)


fir['lat1'] = fir.lat.apply(convert_lat)
fir['lon1'] = fir.lon.apply(convert_lon)
fir['lat2'] = fir.lat1.shift(1)
fir.loc[0, 'lat2'] = fir.loc[fir.index[-1], 'lat1']
fir['lon2'] = fir.lon1.shift(1)
fir.loc[0, 'lon2']= fir.loc[fir.index[-1], 'lon1']


for flightid in flightids:
    ac = df.loc[df['flightid'] == flightid].copy()
    startwpidx = ac.index[0]
    lastwpidx = ac.index[-1]
    callsign = ac.callsign.iloc[0]
    actype = ac.actype.iloc[0]

    for i in ac.index:
        for j in fir.index:
            _, lat, lon = find_intersect(ac.lat1[i], ac.lon1[i], ac.lat2[i], ac.lon2[i],
                           fir.lat1[j], fir.lon1[j], fir.lat2[j], fir.lon2[j])
            if _:
                break

        if _:
            break
    #start interpolating:
    if _:
        qdr, dist = qdrdist(lat, lon, ac.loc[i, 'lat1'], ac.loc[i,'lon1'])

        ratio = dist/ac.loc[i, 'dist']
        df.loc[i, 't1'] = ac.loc[i,'t1'] + datetime.timedelta(seconds=((ac.loc[i,'t2'] - ac.loc[i,'t1']).seconds) * ratio)
        df.loc[i, 'fl1'] = ac.loc[i,'fl1'] + (ac.loc[i,'fl2'] - ac.loc[i,'fl1']) * ratio
        df.loc[i, 'lat1'] = lat
        df.loc[i, 'lon1'] = lon
        df.loc[i, 'sequence'] = 1
        df.drop(np.arange(startwpidx, i), inplace=True)
    else:
        df.drop(np.arange(startwpidx, i+1), inplace=True)



print('Generating trafscript commands...')
# Now sort per aircraft and generate commands
flightids = df.flightid.unique()
aclist=[]

for flightid in flightids:
    ac = df.loc[df['flightid'] == flightid].copy()
    ac['cmd1']=None
    ac['cmd2']=None
    ac['cmd3']=None
    lastwpidx = ac.index[-1]
    callsign = ac.callsign.iloc[0]
    actype = ac.actype.iloc[0]


    for i in ac.index:
        if ac.loc[i, 'sequence'] == 1:
            ac.loc[i, 'cmd1'] = 'CRE {}, {}, {}, {}, {}, {}, {}'.format(callsign,
                  actype, ac.loc[i, 'lat1'], ac.loc[i, 'lon1'], ac.loc[i, 'hdg'], ac.loc[i, 'fl1'], ac.loc[i, 'cas'])
            ac.loc[i, 'cmd2'] = 'ADDWPT {}, {}, {}, {}, {}'.format(callsign,
                  ac.loc[i, 'lat2'], ac.loc[i, 'lon2'], ac.loc[i, 'fl2'],
                  ac.loc[i, 'cas'])

        elif i == lastwpidx:
            ac.loc[i, 'cmd1'] = 'MOVE {}, {}, {}, {} ,{} ,{}, {}'.format(callsign,
                  ac.loc[i, 'lat1'], ac.loc[i, 'lon1'], ac.loc[i, 'fl1'],
                  ac.loc[i, 'hdg'], ac.loc[i, 'cas'], ac.loc[i, 'vs'])
            ac.loc[i, 'cmd2'] = 'ADDWPT {}, {}, {}, {}, {}'.format(callsign,
                  ac.loc[i, 'lat2'], ac.loc[i, 'lon2'], ac.loc[i, 'fl2'],
                  ac.loc[i, 'cas'])
            ac.loc[i, 'cmd3'] = 'DEL {}'.format(callsign)

        else:
            ac.loc[i, 'cmd1'] = 'MOVE {}, {}, {}, {} ,{} ,{}, {}'.format(callsign,
                  ac.loc[i, 'lat1'], ac.loc[i, 'lon1'], ac.loc[i, 'fl1'],
                  ac.loc[i, 'hdg'], ac.loc[i, 'cas'], ac.loc[i, 'vs'])
            ac.loc[i, 'cmd2'] = 'ADDWPT {}, {}, {}, {}, {}'.format(callsign,
                  ac.loc[i, 'lat2'], ac.loc[i, 'lon2'], ac.loc[i, 'fl2'],
                  ac.loc[i, 'cas'])

    aclist.append(ac)

#Now Create three separate dataframes to be merged for all the commands

df = pd.concat(aclist)
#df.sort_values(by=['t1', 't2'], inplace=True)
#df.reset_index(inplace=True)

print('Converting to simtime and sorting...')
df1 = df[['t1', 'cmd1']].rename(index=str, columns={'t1':'t', 'cmd1': 'cmd'})
df2 = df[['t1', 'cmd2']].rename(index=str, columns={'t1':'t', 'cmd2': 'cmd'})
df3 = df[['t2', 'cmd3']].rename(index=str, columns={'t2':'t', 'cmd3': 'cmd'})
df = pd.concat([df1, df2, df3], axis=0).dropna()

df.sort_values(by=['t', 'cmd'], ascending=[True, False], inplace=True)
df.reset_index(inplace=True)

#Create timedelta
df['td'] = df.t - df.loc[0, 't']

#Format time
df.td = df.td.apply(format_time)

output = df.td + df.cmd

print('Writing to file...')
f = open('tempData/' + fn + '.scn', 'w')
for out in output:
    f.write(out+'\n')
f.close()
#output.to_csv('tempData/' + fn + '.scn', index=False, header=False, doublequote=False)


print('Done')


#
#if __name__ == '__main__':
#    fn = '20180705_20180705_0000_2359__EHAM___m3'
#    readFile('tempData/' + fn + '.so6', 1)
#    DatToScn('tempData/' + fn + '.dat')

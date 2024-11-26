#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 28/09/2022
@author: Théa Ragon
"""

from utils import *
PHASENET_API_URL = "https://ai4eps-eqnet.hf.space"

def pickfinder(datadir, ev, st2, arrivals, phasenet, station):
    def process_picks(picks, phase_type, score_threshold):
        """Extract the best pick for a given phase type and score threshold."""
        filtered = picks[picks['phase_type'] == phase_type]
        if not filtered.empty and filtered['phase_score'].max() > score_threshold:
            idx = filtered['phase_score'].idxmax()
            if filtered.at[idx, 'phase_index'] < 10 and len(filtered) > 1:
                idx = filtered['phase_score'].nlargest(2).idxmin()
            return obspy.UTCDateTime(filtered.at[idx, 'phase_time'])
        return None

    def format_time(utc_time):
        """Format UTC time to a string."""
        return utc_time.datetime.strftime("%Y-%m-%dT%H:%M:%S.%f") if utc_time else -1000

    if phasenet:
        picks = []
        try:
            if phasenet == 'api':
                st2.write("data.mseed", format="MSEED")
                data = np.column_stack([tr.data for tr in st2])
                data_id = ev.resource_id.id[-8:]
                _, _, picks_json = gradoclient.predict(
                    ["data.mseed"], json.dumps(data.tolist()), json.dumps([data_id])
                )
                picks = pd.read_json(picks_json)
        except (requests.exceptions.JSONDecodeError, ValueError, subprocess.CalledProcessError):
            pass

        pickP = process_picks(picks, "P", 0.5)
        pickS = process_picks(picks, "S", 0.4)

        ev_id = ev.resource_id.id[-8:]
        arrivals_entry = arrivals[station][ev_id]

        if pickS:
            timeS = (
                st4[2].stats.starttime
                + st4[2].stats.delta * (pickS - st4[2].stats.starttime) / 0.01
                if phasenet == "api" and len(picks) > 1
                else pickS
            )
            arrivals_entry["tS"] = format_time(timeS)
        else:
            arrivals_entry["tS"] = -1000

        if pickP:
            timeP = (
                st4[2].stats.starttime
                + st4[2].stats.delta * (pickP - st4[2].stats.starttime) / 0.01
                if phasenet == "api"
                else pickP
            )
            arrivals_entry["tP"] = format_time(timeP)
            arrivals_entry["delta"] = (
                int(obspy.UTCDateTime(arrivals_entry["tS"]) - obspy.UTCDateTime(arrivals_entry["tP"]))
                if pickS
                else 30
            )
        else:
            arrivals_entry["tP"] = -1000
    else:
        trace = st2[2]
        df = trace.stats.sampling_rate
        cft = classic_sta_lta(trace.data, int(5 * df), int(10 * df))
        onset = trigger_onset(cft, 1.5, 0.5)[0, 0]
        timeP = trace.stats.starttime + trace.times()[onset]
        timeS = timeP + 50

        ev_id = ev.resource_id.id[-8:]
        arrivals_entry = arrivals[station][ev_id]

        arrivals_entry["tP"] = format_time(timeP)
        arrivals_entry["tS"] = format_time(timeS)
        arrivals_entry["delta"] = int(obspy.UTCDateTime(arrivals_entry["tS"]) - obspy.UTCDateTime(arrivals_entry["tP"]))

        return
    
def get_trims(station, evid, arrivals, Swave, delta=None):
    if not Swave:
        tP = arrivals[station].get(evid,{}).get('tP', -1000)
        if delta is None:
            delta = arrivals[station].get(evid, {}).get('delta', 5)
        if tP == -1000: return None, None
        tstart = obspy.UTCDateTime(tP) - 1
        tend = obspy.UTCDateTime(tP) + delta
    else:
        tS = arrivals[station].get(evid, {}).get('tS', -1000)
        if tS == -1000: return None, None
        tstart = obspy.UTCDateTime(tS) - 0.2
        tend = obspy.UTCDateTime(tS) + 13.
    return tstart, tend

# ---- user params
mydir = './'
datadir = mydir+'data'
wfdir = datadir+'/wf'

stpc = False # if True, downloads from SCEDC, else loads locally
phasenet = 'api' # 'api' or False
read_arrivals = True # if True, reads arrivals in dict
Swave = False # if False, use P wave arrivals
use_gc_cat = True # use Zach Ross's catalog. if False use regular SCEDC cat
use_cc = False # use cross-correlation to select EGFs, else distance to main event only.

dist_egf = 0.8 #km
nbr_cc = 4 ## max number of EGFs to select

# ---- Cahuilla specs
lon_ca = [-116.85, -116.75]
lat_ca = [33.45,33.55]
lon_sta = [-119, -114]
lat_sta = [31, 36]
dist_sta = 200 #km
t0_ca = obspy.UTCDateTime(year=2016, month=1, day=1)
t1_ca = obspy.UTCDateTime(year=2019, month=12, day=31)
output_suffix = "_P" if not Swave else "_S"


# ---- Download data
if stpc is True:
    client = Client("SCEDC", timeout=1000)
    catalog_m2 = client.get_events(minlatitude=lat_ca[0], maxlatitude=lat_ca[1],
                                   minlongitude=lon_ca[0], maxlongitude=lon_ca[1],
                                   starttime=t0_ca, endtime=t1_ca,
                                   minmagnitude=2, maxmagnitude=2.5)
    catalog_m2.write(f"{datadir}/cat_m2.xml", format="QUAKEML")
    cat_m4 = client.get_events(minlatitude=lat_ca[0], maxlatitude=lat_ca[1],
                               minlongitude=lon_ca[0], maxlongitude=lon_ca[1],
                               starttime=t0_ca, endtime=t1_ca,
                               minmagnitude=4, maxmagnitude=5)
    cat_m4.write(f"{datadir}/cat_m4.xml", format="QUAKEML")
    main_ev = cat_m4[0]
    inventory = client.get_stations(network="CI",
                                    minlatitude=lat_sta[0], maxlatitude=lat_sta[1],
                                    minlongitude=lon_sta[0], maxlongitude=lon_sta[1],
                                    startbefore=t0_ca, endafter=t1_ca)
    inventory.write(f"{datadir}/stations.xml",format="STATIONXML")

else:
    inventory = obspy.read_inventory(f"{datadir}/stations.xml",format="STATIONXML")
    catalog_m2 = obspy.core.event.catalog.read_events(f"{datadir}/cat_m2.xml", format="QUAKEML")
    main_ev = obspy.core.event.catalog.read_events(f"{datadir}/cat_m4.xml", format="QUAKEML")[0]

# ---- Read Zach's catalog
if use_gc_cat == True:
    fullc = pd.read_csv(f"{datadir}/cat_ross.txt",
                        sep =r"\s+",
                          names=['year', 'month', 'day', 'hour', 'minute', 'second',
                                 'eID', 'latR', 'lonR', 'depR',
                                 'mag', 'qID', 'cID', 'nbranch',
                                 'qnpair', 'qndiffP', 'qndiffS',
                                 'rmsP', 'rmsS',
                                 'eh', 'ez', 'et',
                                 'latC', 'lonC', 'depC'])
    fullc['time'] =  pd.to_datetime(fullc[['year', 'month', 'day', 'hour', 'minute', 'second']], utc=True)
    fullc['time_strf'] = fullc['time'].dt.strftime('%Y-%m-%dT%H:%M')

    main_ev_gc = fullc.loc[fullc['time_strf'] == main_ev.origins[0].time.strftime('%Y-%m-%dT%H:%M')]

# ---- Select M2 events based on distance
cat2 = catalog_m2.filter("longitude <= {}".format(main_ev.origins[0].longitude+dist2lon(lat_ca[0],dist_egf)),
                  "longitude >= {}".format(main_ev.origins[0].longitude-dist2lon(lat_ca[0], dist_egf)),
                  "latitude <= {}".format(main_ev.origins[0].latitude+dist2lat(dist_egf)),
                  "latitude >= {}".format(main_ev.origins[0].latitude-dist2lat(dist_egf)) )
egf_ev = cat2

if use_gc_cat:
    cat2_gc = fullc.loc[(fullc['lonR'] <= main_ev_gc.lonR.values[0] + dist2lon(lat_ca[0], dist_egf))
                        & (fullc['lonR'] >= main_ev_gc.lonR.values[0] - dist2lon(lat_ca[0], dist_egf))
                        & (fullc['latR'] <= main_ev_gc.latR.values[0] + dist2lat(dist_egf))
                        & (fullc['latR'] >= main_ev_gc.latR.values[0] - dist2lat(dist_egf))]

    # reassign events to catalog
    ev_match = []
    for i in range(len(cat2_gc)):
        event_time = cat2_gc.iloc[i]['time']
        start_time = (event_time - pd.Timedelta(seconds=30)).strftime('%Y-%m-%dT%H:%M')
        end_time = (event_time + pd.Timedelta(seconds=60)).strftime('%Y-%m-%dT%H:%M')
        matching_events = catalog_m2.filter(f"time >= {start_time}", f"time <= {end_time}")

        if matching_events and matching_events[0].resource_id.id[-8:] not in {
            ev.resource_id.id[-8:] for ev in ev_match}:
            ev_match.append(matching_events[0])
    egf_ev = ev_match


    
# ---- Select stations
stations = []
for j in range(len(inventory[0])):
    stlo = inventory[0].stations[j].longitude
    stla = inventory[0].stations[j].latitude

    di = haversine(main_ev.origins[0].longitude, main_ev.origins[0].latitude, stlo, stla)

    if di <= dist_sta:
        stations.append(inventory[0].stations[j].code)

# For the sake of the example, selecting only 4 stations
if use_cc:
    stations = ['BBS', 'CTW','BLA2', 'PSD']
else:
    stations = ['BOR', 'CTW', 'BLA2', 'PSD']

# ---- Download WF
if stpc is True:
    main_wf = obspy.Stream()
    main_wf = client.get_waveforms(network='CI,PB,AZ,YN,SB', station=','.join(stations),
                                    location='*', channel='BH?',
                                    starttime=main_ev.origins[0].time - 30,
                                    endtime=main_ev.origins[0].time + 70)
    main_wf.write(f"{wfdir}/main_wf.mseed", format="mseed")
    
    eg_wf = {}
    for ev in egf_ev:
        eg_wf[ev.resource_id.id[-8:]] = client.get_waveforms(network='CI,PB,AZ,YN,SB',station = ','.join(stations),
                            location='*',channel='BH?',
                            starttime= ev.origins[0].time - 30, endtime = ev.origins[0].time + 70)
        eg_wf[ev.resource_id.id[-8:]].write(f"{wfdir}/{ev.resource_id.id[-8:]}_wf.mseed", format="mseed")
else:
    eg_wf = {}
    print('Reading waveforms')
    for ev in egf_ev:
        eg_wf[ev.resource_id.id[-8:]] = obspy.Stream()
        eg_wf[ev.resource_id.id[-8:]] = obspy.read(f"{wfdir}/{ev.resource_id.id[-8:]}_wf.mseed", format="mseed")
    main_wf = obspy.read(f"{wfdir}/main_wf.mseed", format="mseed")

# ---- Find P and S arrivals
if read_arrivals:
    with open(f"{datadir}/arrivals.json", 'r') as f:
        arrivals = json.load(f)
else:
    print('Calculating phase arrivals')
    if phasenet == 'api':
        from gradio_client import Client as gradioclient
        gradoclient = gradioclient("ai4eps/phasenet", serialize=True)

    try:
        with open(f"{datadir}/arrivals.json", 'r') as f:
            arrivals = json.load(f)
    except FileNotFoundError:
        arrivals = {}

    for station in stations:
        st4 = main_wf.select(station=station)
        st4.sort()
        if len(st4) == 0: continue

        station_arrivals = arrivals.setdefault(station, {})
        main_event_id = main_ev.resource_id.id[-8:]
        event_arrivals = station_arrivals.setdefault(main_event_id, {})

        # Calculate picks for main event if not already available
        if 'tS' not in event_arrivals or 'tP' not in event_arrivals:
            pickfinder(datadir, main_ev, st4, arrivals, phasenet, station)

        if event_arrivals.get('tS', -1000) == -1000 or event_arrivals.get('tP', -1000) == -1000:
            continue

        # Process each EGF event
        for ev in egf_ev:
            egf_event_id = ev.resource_id.id[-8:]
            if egf_event_id not in station_arrivals:
                st2 = eg_wf.get(egf_event_id, []).select(station=station)
                if len(st2) == 0: continue

                # Preselection based on SNR
                snr = SNR(st2[0].data, st2[0].data[:int(30 / st2[0].stats.delta)])
                if snr < 1.0: continue
                
                station_arrivals[egf_event_id] = {}
                pickfinder(datadir, ev, st2, arrivals, phasenet, station)

    # Save updated arrivals to file
    with open(f"{datadir}/arrivals.json", 'w') as f:
        json.dump(arrivals, f)


# ---- Select M2 events based on cross-correlation of P  arrivals at each station

if use_cc:
    print('Selecting M2 events based on CC')
    specs_sta = {}

    for station in stations:
        specs_sta[station] = {}
        st4 = main_wf.select(station=station)
        if len(st4) == 0: continue

        tstart, tend = get_trims(station, main_ev.resource_id.id[-8:], arrivals, Swave)

        st4_P = st4.copy()
        st4_P.trim(tstart, tend)

        if len(st4_P) < 2: continue

        if st4_P[0].stats.sampling_rate >= 20.0:
            for tr in st4_P:
                tr.decimate(factor=2, strict_length=False)

        snr = []
        for ev in egf_ev:
            st2 = eg_wf[ev.resource_id.id[-8:]].select(station=station)
            if len(st2) < 2: continue
            snr_value = SNR(st2[0].data, st2[0].data[:int(20 / st2[0].stats.delta)])
            if snr_value < 1.0: continue
            snr.append(snr_value)

        if not snr or max(snr) < 1.0: continue
        idx_snr = np.array(snr).argsort()[-50:][::-1]
        ev_egf_snr = [egf_ev[i] for i in idx_snr]

        cc, cc_shift = [], []
        for ev in ev_egf_snr:
            st2 = eg_wf[ev.resource_id.id[-8:]].select(station=station)
            if len(st2) < 2: continue

            for tr in st2:
                tr.detrend(type='demean')

            tstart, tend = get_trims(station, ev.resource_id.id[-8:], arrivals,
                                     Swave, delta = arrivals[station][main_ev.resource_id.id[-8:]].get('delta'))

            st2_P = st2.copy()

            if tstart is None:
                continue
            st2_P.trim(tstart, tend)

            if len(st2_P) < 2 or len(st2_P[0]) < len(st4_P[0]): continue

            if st2_P[0].stats.sampling_rate >= 20.0:
                for tr in st2_P:
                    tr.decimate(factor=2, strict_length=False)

            cco = [ obspy.signal.cross_correlation.correlate(
                    st2_P[k].data / np.amax(np.abs(st2_P[k].data)),
                    st4_P[k].data / np.amax(np.abs(st4_P[k].data)),
                    shift=len(st2_P[0].data) // 2 + 1) for k in range(3) ]

            ccmax = [obspy.signal.cross_correlation.xcorr_max(cco[i])[1] for i in range(3)]
            shift = [obspy.signal.cross_correlation.xcorr_max(cco[i])[0] for i in range(3)]
            cc.append(np.mean(ccmax))
            cc_shift.append(np.array(np.unique(shift, return_counts=True))[:,0])

        idx_cc = [i for i in np.array(cc).argsort()[::-1] if cc[i] > 0.2][-nbr_cc:]
        ccsta_list = [ev_egf_snr[i].resource_id.id[-8:] for i in idx_cc]
        ccsta_shift = [cc_shift[i][0] if cc_shift[i][1] >= 2 else 0 for i in idx_cc]

        specs_sta[station]['id'] = ccsta_list
        specs_sta[station]['cc'] = np.array(cc)[idx_cc].tolist()
        specs_sta[station]['shift'] = [int(x) for x in ccsta_shift]
else:
    specs_sta = {}
    dist = np.array(
        [haversine(main_ev.origins[0].longitude, main_ev.origins[0].latitude, ev0.origins[0].longitude, ev0.origins[0].latitude)
         for ev0 in egf_ev])
    dist_idx = dist.argsort()[:20]
    idx = [egf_ev[x].resource_id.id[-8:] for x in dist_idx]
    for station in stations:
        specs_sta[station] = {}
        specs_sta[station]['id'] = idx
        specs_sta[station]['cc'] = [1]*20
        specs_sta[station]['shift'] = [0]*20

with open(f"{datadir}/cc_specs.json", 'w') as fp:
    json.dump(specs_sta, fp)

# ---- Select WF
len_egf = {}
stations_wd_idx = []

ids = np.array([ev.resource_id.id[-8:] for ev in egf_ev])
for station in stations:
    station_ok = False
    try:
        idx_egf_ev = [ np.where(ids == specs_sta[station]['id'][k])[0][0]
            for k in range(len(specs_sta[station]['id']))]
        egf_ev_sta = [egf_ev[i] for i in idx_egf_ev][:nbr_cc]
    except KeyError:
        continue
    st_all, st_all_trim = obspy.Stream(), obspy.Stream()

    for i, ev in enumerate(egf_ev_sta):
        st = eg_wf[ev.resource_id.id[-8:]].select(station=station)
        if len(st) == 0: continue
        station_ok = True

        # ev.write(f"{datadir}/{ev.resource_id.id[-8:]}.xml", format="QUAKEML")
        for tr in st:
            tr.detrend(type='simple')
            st_all.append(tr)

        tstart, tend = get_trims(station, ev.resource_id.id[-8:], arrivals,
                                 Swave, delta = arrivals[station][main_ev.resource_id.id[-8:]]['delta'])

        try:
            tstart += specs_sta[station]['shift'][i]
            tend += specs_sta[station]['shift'][i]
            st_trim = st.copy().trim(tstart, tend)

            if st_trim[0].stats.sampling_rate >= 20.0:
                for tr in st_trim:
                    tr.decimate(factor=2, strict_length=False)

            max_val = max(np.amax(np.abs(tr.data)) for tr in st_trim)
            for tr in st_trim:
                tr.data /= max_val
                st_all_trim.append(tr)
        except:
            continue

    # if len(st_all) != 0: st_all.write(f"{datadir}/{station}_multi_gf.mseed")
    if len(st_all_trim) == 0: station_ok = False

    if station_ok:
        # main_ev.write(f"{datadir}/{main_ev.resource_id.id[-8:]}.xml", format="QUAKEML")

        st = main_wf.select(station=station)
        if len(st) == 0:
            station_ok = False
            break

        for tr in st:
            tr.detrend(type='simple')

        tstart, tend = get_trims(station, main_ev.resource_id.id[-8:], arrivals, Swave)
        st_trim = st.copy().trim(tstart, tend)
        if st_trim[0].stats.sampling_rate >= 20.0:
            for tr in st_trim:
                tr.decimate(factor=2, strict_length=False)
        st_trim.write(f"{datadir}/{station}_trc{output_suffix}.mseed")

        ref_len = len(st_trim[0])
        for tr in st_all_trim:
            if len(tr) != ref_len:
                st_all_trim.remove(tr)

        if len(st_all_trim)!=0:
            st_all_trim.write(f"{datadir}/{station}_multi_gf{output_suffix}.mseed")
            stations_wd_idx.append(stations.index(station))
            len_egf[station] = len(st_all_trim)

with open(f"{datadir}/len_egf{output_suffix}.json", 'w') as f:
    json.dump(len_egf, f)

sta_list = [stations[i] for i in stations_wd_idx]
np.save(f"{datadir}/station_list{output_suffix}.npy", sta_list)


## TODO!!
# ---- Write DeepGEM launch file
launcher = writeDeepGEMlauncher(datadir,
                                './out',
                                 stations,
                                 num_egf = [str(x//3) for x in list(len_egf.values())] )
print('DeepGEM launch file written in {}'.format(launcher))
print("use './EGF_ex.sh path_to_DeepGEM' to launch")


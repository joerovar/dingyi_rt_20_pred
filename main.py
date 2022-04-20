import numpy as np
import matplotlib.pyplot as plt


odt_ground_5 = np.load('joseph_ground_true.npy')
odt_stnb_5 = np.load('joseph_stnb.npy')
idx_stops = np.load('joseph_select_area.npy')
non_zero_idx = np.load('joseph_flatten_index.npy')


def increase_interval_size(odt, prev_interval_min=5, new_interval_min=30):
    ratio_interval = int(new_interval_min/prev_interval_min)
    od_pair_len = odt.shape[1]
    clip_nr = int(odt.shape[0] % ratio_interval)
    if clip_nr:
        odt_clipped = odt[:-clip_nr]
    else:
        odt_clipped = odt
    odt_grouped = odt_clipped.reshape((-1, ratio_interval, od_pair_len))
    odt_new = np.sum(odt_grouped, axis=-2)
    return odt_new


def interval_average(odt, interval_min):
    nr_intervals_day = int(60*24/interval_min)
    od_pair_len = odt.shape[1]
    clip_nr = int(odt.shape[0] % nr_intervals_day)
    if clip_nr:
        odt_clipped = odt[:-clip_nr]
    else:
        odt_clipped = odt
    odt_grouped = odt_clipped.reshape((-1, nr_intervals_day, od_pair_len))
    tot_boards_all_intervals = np.sum(odt_grouped, axis=-1)
    boards_per_interval = np.average(tot_boards_all_intervals, axis=-2)
    return boards_per_interval


def extract_weekday_odt(odt, interval_min, first_monday_idx=1):
    nr_intervals_day = int(60 * 24 / interval_min)
    od_pair_len = odt.shape[1]
    clip_nr = int(odt.shape[0] % nr_intervals_day)
    if clip_nr:
        odt_clipped = odt[:-clip_nr]
    else:
        odt_clipped = odt
    odt_grouped = odt_clipped.reshape((-1, nr_intervals_day, od_pair_len))
    nr_days = odt_grouped.shape[0]
    mondays_idx = np.arange(first_monday_idx, nr_days, 7)
    weekdays_idx = [[i for i in range(mi, mi+5)] for mi in mondays_idx]
    # clip Labor day Sep 2, 2019
    weekdays_idx = np.array(weekdays_idx).flatten()[1:]
    odt_all_weekdays = odt_grouped[weekdays_idx, :, :]
    odt_avg_weekday = np.average(odt_all_weekdays, axis=-3)
    return odt_avg_weekday


def bar_chart_comparison(obs, pred, fname, x_y_lbls):
    w = 0.5
    x = np.arange(1, obs.shape[0]+1)
    plt.bar(x-w/2, obs, width=w, label='observed')
    plt.bar(x+w/2, pred, width=w, label='predicted')
    plt.xticks(np.arange(1, obs.shape[0]+1, 2))
    plt.ylabel(x_y_lbls[1])
    plt.xlabel(x_y_lbls[0])
    plt.legend()
    plt.savefig(fname)
    plt.close()
    return


odt_ground_day = increase_interval_size(odt_ground_5, prev_interval_min=5, new_interval_min=60*24)
odt_stnb_day = increase_interval_size(odt_stnb_5, prev_interval_min=5, new_interval_min=60*24)

tot_board_ground_day = odt_ground_day.sum(axis=-1)
tot_board_stnb_day = odt_stnb_day.sum(axis=-1)
bar_chart_comparison(tot_board_ground_day, tot_board_stnb_day, 'daily_demand.png',
                     ['day of month', 'tot pax'])


odt_ground_5_weekday = extract_weekday_odt(odt_ground_5, 5) # single day intervals (averaged over all weekdays)
odt_stnb_5_weekday = extract_weekday_odt(odt_stnb_5, 5)

odt_ground_60_weekday = increase_interval_size(odt_ground_5_weekday, prev_interval_min=5, new_interval_min=60)
odt_stnb_60_weekday = increase_interval_size(odt_stnb_5_weekday, prev_interval_min=5, new_interval_min=60)

ground_boards_per_inter = interval_average(odt_ground_60_weekday, interval_min=60)
stnb_boards_per_inter = interval_average(odt_stnb_60_weekday, interval_min=60)
bar_chart_comparison(ground_boards_per_inter, stnb_boards_per_inter, 'hourly_demand_weekday.png',
                     ['hour of day', 'tot pax'])

odt_ground_30_weekday = increase_interval_size(odt_ground_5_weekday, prev_interval_min=5, new_interval_min=30)
odt_stnb_30_weekday = increase_interval_size(odt_stnb_5_weekday, prev_interval_min=5, new_interval_min=30)


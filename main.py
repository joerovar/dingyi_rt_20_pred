import numpy as np
import matplotlib.pyplot as plt


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


odt_ground_5 = np.load('joseph_ground_true.npy')
odt_stnb_5 = np.load('joseph_stnb.npy')
stop_ids = np.load('joseph_select_area.npy')
non_zero_idx = np.load('joseph_flatten_index.npy')

day_3_4_tot = np.sum(odt_ground_5[:288*3], axis=-1)
print(day_3_4_tot.shape)
plt.plot(np.arange(len(day_3_4_tot)), day_3_4_tot)
plt.show()
plt.close()
odt_ground_day = increase_interval_size(odt_ground_5, prev_interval_min=5, new_interval_min=60*24)
odt_stnb_day = increase_interval_size(odt_stnb_5, prev_interval_min=5, new_interval_min=60*24)

tot_board_ground_day = odt_ground_day.sum(axis=-1)
tot_board_stnb_day = odt_stnb_day.sum(axis=-1)
# bar_chart_comparison(tot_board_ground_day, tot_board_stnb_day, 'daily_demand.png',
#                      ['day of month', 'tot pax'])


odt_ground_5_weekday = extract_weekday_odt(odt_ground_5, 5) # single day intervals (averaged over all weekdays)
odt_stnb_5_weekday = extract_weekday_odt(odt_stnb_5, 5)

# tot_5_weekday = np.sum(odt_ground_5_weekday, axis=-1)
# plt.plot(np.arange(len(tot_5_weekday)), tot_5_weekday)
# plt.show()
# plt.close()
odt_ground_60_weekday = increase_interval_size(odt_ground_5_weekday, prev_interval_min=5, new_interval_min=60)
odt_stnb_60_weekday = increase_interval_size(odt_stnb_5_weekday, prev_interval_min=5, new_interval_min=60)

ground_boards_per_inter = interval_average(odt_ground_60_weekday, interval_min=60)
stnb_boards_per_inter = interval_average(odt_stnb_60_weekday, interval_min=60)
# bar_chart_comparison(ground_boards_per_inter, stnb_boards_per_inter, 'hourly_demand_weekday.png',
#                      ['hour of day', 'tot pax'])

nz_odt_ground_30_wkday = increase_interval_size(odt_ground_5_weekday, prev_interval_min=5, new_interval_min=30)
nz_odt_stnb_30_wkday = increase_interval_size(odt_stnb_5_weekday, prev_interval_min=5, new_interval_min=30)

nr_stops = stop_ids.shape[0]
odt_rates_true_30_wkday = np.zeros(shape=(nz_odt_ground_30_wkday.shape[0], nr_stops*nr_stops))
odt_rates_true_30_wkday[:, non_zero_idx] = nz_odt_ground_30_wkday*2
odt_rates_true_30_wkday = np.reshape(odt_rates_true_30_wkday, (odt_rates_true_30_wkday.shape[0], nr_stops, nr_stops))

odt_rates_stnb_30_wkday = np.zeros(shape=(nz_odt_stnb_30_wkday.shape[0], nr_stops*nr_stops))
odt_rates_stnb_30_wkday[:, non_zero_idx] = nz_odt_stnb_30_wkday*2
odt_rates_stnb_30_wkday = np.reshape(odt_rates_stnb_30_wkday, (odt_rates_stnb_30_wkday.shape[0], nr_stops, nr_stops))

# np.save('rt_20_odt_rates_30.npy', odt_rates_stnb_30_wkday)
# np.save('rt_20_demand_stops.npy', stop_ids)

# outbound_stops = np.load('rt_20_out_stops.pkl', allow_pickle=True)
# arr_rates = np.sum(odt_rates_stnb_30_wkday, axis=-1)
# stop_ids_lst = list(stop_ids)
# idx_outbound = [stop_ids_lst.index(int(s)) for s in outbound_stops]
# outbound_arr_rates = arr_rates[2:32, idx_outbound]
# print(np.sum(outbound_arr_rates, axis=-1))

# plt.imshow(odt_ground_30_wkday[6*2], cmap='Greys')
# plt.colorbar()
# plt.show()
# plt.close()
# plt.imshow(odt_ground_30_wkday[14*2], cmap='Greys')
# plt.colorbar()
# plt.show()
# plt.close()


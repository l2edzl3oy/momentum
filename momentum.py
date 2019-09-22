import json
import requests
import time
import math
import pandas.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def init_config(config_file_path="config.json"):
    with open(config_file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def download_data(base_url, symbol, api_key, file_path):
    r = requests.get(base_url.format(symbol, api_key))
    with open(file_path, "wb") as f:
        f.write(r.content)
        print("Downloaded '{}' to '{}'.".format(symbol, file_path))


def add_ewm(df, symbol):
    """
    If there are missing "close" values, EWM will just copy the previous EWM value.

    :param df:
    :param symbol:
    :return:
    """
    print("Adding 48w and 12w EWM for '{}'...".format(symbol))
    col_name = "{}_close".format(symbol)
    close_ewm_48w = df[col_name].ewm(span=48, min_periods=48).mean()
    close_ewm_12w = df[col_name].ewm(span=12, min_periods=12).mean()
    df["{}_ewm_48w".format(symbol)] = close_ewm_48w
    df["{}_ewm_12w".format(symbol)] = close_ewm_12w
    return df


def add_upr(df, symbol):
    """
    Upside Potential Ratio.

    :param df:
    :param symbol:
    :return:
    """
    print("Adding UPR for '{}'...".format(symbol))

    valid_index_start = max(df["{}_open".format(symbol)].first_valid_index(),
                            df["{}_close".format(symbol)].first_valid_index())
    valid_index_end = min(df["{}_open".format(symbol)].last_valid_index(),
                          df["{}_close".format(symbol)].last_valid_index())
    lookback_wks = 12

    for i in range(valid_index_start + lookback_wks, valid_index_end + 2):
        ret_sort = pd.DataFrame()

        ret_sort["open"] = df["{}_open".format(symbol)].iloc[i - lookback_wks:i]
        ret_sort["close"] = df["{}_close".format(symbol)].iloc[i - lookback_wks:i]
        ret_sort["return"] = (ret_sort["close"] - ret_sort["open"]) / ret_sort["open"]

        ret_sort["return_pos"] = ret_sort["return"]
        ret_sort["return_pos"][ret_sort["return_pos"] <= 0.0] = 0.0
        upside = ret_sort["return_pos"].mean()

        ret_sort["return_neg"] = ret_sort["return"]
        ret_sort["return_neg"][ret_sort["return_neg"] >= 0.0] = 0.0
        ret_sort["return_neg_sq"] = ret_sort["return_neg"] ** 2
        downside = ret_sort["return_neg_sq"].mean()
        downside = downside ** (1 / 2)
        if downside <= 0:
            downside = 0.1
            print("Replaced '0' downside with '{}' for {}.".format(0.1, symbol))

        upr = min(upside / downside, 3)

        df.at[i - 1, "{}_upr".format(symbol)] = upr

    return df


def add_signal(df, symbol):
    print("Adding signal for '{}'...".format(symbol))

    df_upr = df["{}_upr".format(symbol)]

    total_upr_len = len(df_upr[df_upr >= -1])
    threshold_upr_len = len(df_upr[df_upr >= 1])
    print("UPR >= 1: {0:.2f}%".format(threshold_upr_len / total_upr_len * 100))

    upr_mean = df_upr.mean()
    print("Mean: {0:.4f}".format(upr_mean))

    upr_median = df_upr.median()
    print("Median: {0:.4f}".format(upr_median))

    upr_quantiles = df_upr.quantile(np.arange(0, 1.01, 0.25))
    print("UPR quantiles:\n{}".format(upr_quantiles))

    start_i = df["{}_ewm_48w".format(symbol)].first_valid_index()
    end_i = df["{}_ewm_48w".format(symbol)].last_valid_index()

    for i in range(start_i, end_i + 1):
        upr = df_upr.loc[i]
        ewm_12w = df["{}_ewm_12w".format(symbol)].loc[i]
        ewm_48w = df["{}_ewm_48w".format(symbol)].loc[i]

        if pd.isna(upr) or pd.isna(ewm_12w) or pd.isna(ewm_48w):
            print("Invalid values at index '{}' for '{}'. "
                  "upr: {}. ewm_12w: {}. ewm_48w: {}.".format(i, symbol, upr, ewm_12w, ewm_48w))
            df.at[i, "{}_signal".format(symbol)] = math.nan
        else:
            if upr <= upr_quantiles.iloc[1] and ewm_12w <= ewm_48w:
                diff_series = df_upr.loc[i - 3:i]
                if diff_series.isna().values.any():
                    print("Invalid values at index '{}' for '{}'. diff_series: {}.".format(i, symbol, diff_series))
                    df.at[i, "{}_signal".format(symbol)] = math.nan
                else:
                    diff_series_below_days = len(diff_series[diff_series <= upr_quantiles.iloc[1]]) / 4 * 100
                    upr_mean_1 = df_upr.loc[i - 3:i - 2].mean()
                    upr_mean_2 = df_upr.loc[i - 1:i].mean()
                    if diff_series_below_days >= 100.0 and (upr_mean_1 >= upr_mean_2):
                        df.at[i, "{}_signal".format(symbol)] = -1
                    else:
                        df.at[i, "{}_signal".format(symbol)] = -0.5
            elif upr >= upr_quantiles.iloc[3] and ewm_12w >= ewm_48w:
                diff_series = df_upr.loc[i - 3:i]
                if diff_series.isna().values.any():
                    print("Invalid values at index '{}' for '{}'. diff_series: {}.".format(i, symbol, diff_series))
                    df.at[i, "{}_signal".format(symbol)] = math.nan
                else:
                    diff_series_above_days = len(diff_series[diff_series >= upr_quantiles.iloc[3]]) / 4 * 100
                    upr_mean_1 = df_upr.loc[i - 3:i - 2].mean()
                    upr_mean_2 = df_upr.loc[i - 1:i].mean()
                    if diff_series_above_days >= 100.0 and (upr_mean_1 <= upr_mean_2):
                        df.at[i, "{}_signal".format(symbol)] = 1
                    else:
                        df.at[i, "{}_signal".format(symbol)] = 0.5
            else:
                df.at[i, "{}_signal".format(symbol)] = 0

    return df


def colour_upr(df, symbols):
    plt.close("all")

    line_cmap = plt.get_cmap('RdYlGn', 1024)
    light_grey = matplotlib.colors.to_rgba("lightgrey")

    fig, axs = plt.subplots(2, 1)
    fig.set_figwidth(25)
    fig.set_figheight(len(symbols) + 1)

    axs[0].axis("tight")
    axs[0].axis("off")

    df_len = len(df)

    data = list()
    for symbol in symbols:
        data.append([round(upr, 4) for upr in df["{}_upr".format(symbol)].iloc[df_len - 24:].tolist()])

    plt.tight_layout()

    # Build colours
    colour_range = np.linspace(0.15, 0.85, len(data))
    colours = [[] for _ in enumerate(data)]
    for i, _ in enumerate(data[0]):
        comp_list = [0.0 if math.isnan(data[j][i]) else data[j][i] for j in range(0, len(data))]
        comp_list_sorted_idx = np.argsort(comp_list).tolist()
        # print(comp_list, comp_list_sorted_idx)
        for k, _ in enumerate(data):
            if math.isnan(data[k][i]):
                colours[k].append(light_grey)
            else:
                colours[k].append(line_cmap(colour_range[comp_list_sorted_idx.index(k)]))

    columns = df["timestamp"].iloc[df_len - 24:].dt.date.tolist()

    table = axs[0].table(cellText=data, cellColours=colours, rowLabels=symbols, colLabels=columns, loc='center')
    table.scale(1, 1.5)
    table.auto_set_column_width([i for i in range(0, len(columns))])

    for symbol in symbols:
        axs[1].plot(df["timestamp"].iloc[df_len - 24:], df["{}_upr".format(symbol)].tail(24),
                    label="{}_upr".format(symbol))

    axs[1].xaxis.set_ticks(columns)
    axs[1].xaxis.set_tick_params(rotation=45)
    axs[1].legend()
    axs[1].axis("tight")

    plt.tight_layout()

    # plt.show()
    plt.savefig("data/joined_df.png")


if __name__ == "__main__":
    pandas.plotting.register_matplotlib_converters()

    config = init_config()
    print("Loaded config: {}".format(config))

    joined_df = None

    for config_symbol in config["symbols"]:
        print("-----'{}' start-----".format(config_symbol))
        config_file_path = "data/{}.csv".format(config_symbol.replace(".", "_"))
        start_time = time.time()

        if config["download_data"]:
            print("Downloading {} of {}: {}...".format(config["symbols"].index(config_symbol) + 1,
                                                       len(config["symbols"]), config_symbol))
            download_data(config["base_url"], config_symbol, config["api_key"], config_file_path)

        print("Pre-processing '{}'...".format(config_symbol))
        temp_df = pd.read_csv(config_file_path)

        # Drop unnecessary columns. Pre-pend symbol in col name.
        temp_df = temp_df.drop(columns=["high", "low", "volume"])
        col_names = ["{}_{}".format(config_symbol, col_name) for col_name in temp_df.columns if col_name != "timestamp"]
        col_names.insert(0, "timestamp")
        temp_df.columns = col_names

        # Reverse so that past -> present time.
        temp_df = temp_df[::-1].reset_index(drop=True)

        # Remove zeros.
        open_col_name = "{}_open".format(config_symbol)
        temp_df.loc[temp_df[open_col_name] == 0, open_col_name] = math.nan
        close_col_name = "{}_close".format(config_symbol)
        temp_df.loc[temp_df[close_col_name] == 0, close_col_name] = math.nan

        # Add statistics.
        temp_df = add_ewm(temp_df, config_symbol)
        temp_df = add_upr(temp_df, config_symbol)
        temp_df = add_signal(temp_df, config_symbol)

        if joined_df is None:
            joined_df = temp_df
        else:
            joined_df = joined_df.join(temp_df.set_index("timestamp"), on="timestamp", how='outer')
        print("Pre-processed '{}'.".format(config_symbol))

        elapsed_time = time.time() - start_time

        if config["download_data"] and elapsed_time < 11:
            time_to_sleep = math.ceil(11 - elapsed_time)
            print("Rate-limiting. '{}' sec sleep...".format(time_to_sleep))
            time.sleep(time_to_sleep)

        print("-----'{}' end-----".format(config_symbol))

    joined_df["timestamp"] = pd.to_datetime(joined_df["timestamp"])
    joined_df = joined_df.sort_values(by=["timestamp"])
    joined_df = joined_df.reset_index(drop=True)

    colour_upr(joined_df, config["symbols"])

    joined_df.to_csv("data/joined_df.csv")
    print("Saved CSV.")

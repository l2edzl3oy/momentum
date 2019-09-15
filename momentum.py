import json
import requests
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def init_config(config_file_path="config.json"):
    with open(config_file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def download_data(base_url, symbol, api_key, file_path):
    r = requests.get(base_url.format(symbol, api_key))
    with open(file_path, "wb") as f:
        f.write(r.content)
        print("Downloaded '{}' to '{}'.".format(symbol, file_path))


def calculate_upr(df, symbol):
    """
    Upside Potential Ratio.

    :param df:
    :param symbol:
    :return:
    """

    valid_index_start = df["{}_open".format(symbol)].first_valid_index()
    valid_index_end = df["{}_open".format(symbol)].last_valid_index()
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
        if downside == 0:
            downside = df["{}_upr".format(symbol)].min()

        upr = min(upside / downside, 3)

        df.at[i - 1, "{}_upr".format(symbol)] = upr

    return df


def colour_upr(df, symbols):
    plt.close("all")

    line_cmap = plt.get_cmap('RdYlGn', 1024)

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

    # plt.subplots_adjust(left=0.1)

    # plt.show()
    plt.savefig("data/df.png")


if __name__ == "__main__":
    config = init_config()
    print("Loaded config: {}".format(config))

    df = None

    for symbol in config["symbols"]:
        print("Downloading {} of {}: {}...".format(config["symbols"].index(symbol) + 1, len(config["symbols"]), symbol))
        file_path = "data/{}.csv".format(symbol.replace(".", "_"))

        start_time = time.time()
        # download_data(config["base_url"], symbol, config["api_key"], file_path)

        print("Pre-processing '{}'...".format(symbol))
        temp_df = pd.read_csv(file_path)
        # Remove zeros.
        temp_df = temp_df[(temp_df["open"] > 0.0) & (temp_df["close"] > 0.0)]
        # Drop unnecessary columns.
        temp_df = temp_df.drop(columns=["high", "low", "volume"])
        col_names = ["{}_{}".format(symbol, col_name) for col_name in temp_df.columns if col_name != "timestamp"]
        col_names.insert(0, "timestamp")
        temp_df.columns = col_names

        if df is None:
            df = temp_df
        else:
            df = df.join(temp_df.set_index("timestamp"), on="timestamp", how='outer')
        print("Pre-processed '{}'.".format(symbol))

        elapsed_time = time.time() - start_time
        if elapsed_time < 11:
            time_to_sleep = math.ceil(11 - elapsed_time)
            print("Rate-limiting. '{}' sec sleep...".format(time_to_sleep))
            # time.sleep(time_to_sleep)

    df = df.dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["timestamp"])
    df = df.reset_index(drop=True)

    for symbol in config["symbols"]:
        df = calculate_upr(df, symbol)

    df.to_csv("data/df.csv")

    colour_upr(df, config["symbols"])

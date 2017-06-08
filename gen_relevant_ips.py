from configures import *
from data_view import DataView


def ip_group(df):
    group_num = -1
    grouped_ip = {}
    grouped_num = {}
    i = 0
    while i < len(df):
        flag = True
        for k in grouped_ip.keys():
            v = grouped_ip[k]
            if df.at[i, src_ip] in v:
                v.add(df.at[i, des_ip])
                grouped_num[k].append(i)
                flag = False
            elif df.at[i, des_ip] in v:
                v.add(df.at[i, src_ip])
                grouped_num[k].append(i)
                flag = False
        if flag:
            group_num += 1
            grouped_ip[group_num] = {df.at[i, src_ip], df.at[i, des_ip]}
            grouped_num[group_num] = [i]
            print(len(grouped_ip))
        i += 1
    return grouped_num, grouped_ip


def save_csv(df, groups, i):
    df.loc[groups[i]][[order_num, occur_time, src_ip, des_ip, src_port, des_port, theme, security_type]].\
        to_csv(relevant_ip_path + str(i) + '.csv', index=False)


def gen_ips_csv():
    dv = DataView()
    df = dv.df
    ip_groups, ip_groups2 = ip_group(df)
    for k, v in ip_groups.items():
        save_csv(df, ip_groups, k)


if __name__ == '__main__':
    gen_ips_csv()

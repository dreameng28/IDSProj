import pandas as pd
from configures import *


class DataView:
    def __init__(self, file_path=csv_file_path):
        df = pd.read_csv(file_path, encoding='gbk')
        self.df = df

    @property
    def type_list(self):
        return self.df[security_type].tolist()

    @property
    def degree_list(self):
        return self.df[event_degree].tolist()

    @property
    def time_list(self):
        return self.df[occur_time].tolist()

    @property
    def src_ip_list(self):
        return self.df[src_ip].tolist()

    @property
    def des_ip_list(self):
        return self.df[des_ip].tolist()

    @property
    def src_port_list(self):
        return self.df[src_port].tolist()

    @property
    def des_port_list(self):
        return self.df[des_port].tolist()

    @property
    def response_list(self):
        return self.df[event_response].tolist()

    @property
    def theme_set(self):
        return set(self.df[theme].tolist())

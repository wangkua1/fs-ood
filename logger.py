import json

import numpy as np


def extract_meter_values(meters, std=False):
    ret = { }

    for split in meters.keys():
        ret[split] = { }
        for field,meter in meters[split].items():
            if std:
                ret[split][field] = meter.value()
            else:
                ret[split][field] = meter.value()[0]

    return ret


def render_meter_values(meter_values, N=1):
    field_info = []
    for split in meter_values.keys():
        for field,val in meter_values[split].items():
            if type(val) is tuple:
                field_info.append("{:s} {:s} = ({:0.3f},{:0.3f})".format(split, field, val[0], 1.96 * val[1] / N))
            else:   
                field_info.append("{:s} {:s} = {:0.6f}".format(split, field, val))

    return ', '.join(field_info)


def convert_array(d):
    ret = { }
    for k,v in d.items():
        if isinstance(v, dict):
            ret[k] = { }
            for kk,vv in v.items():
                ret[k][kk] = np.array(vv)
        else:
            ret[k] = np.array(v)

    return ret


def load_trace(trace_file):
    ret = { }

    with open(trace_file, 'r') as f:
        for i,line in enumerate(f):
            vals = json.loads(line.rstrip('\n'))

            if i == 0:
                for k,v in vals.items():
                    if isinstance(v, dict):
                        ret[k] = { }
                        for kk in v.keys():
                            ret[k][kk] = []
                    else:
                        ret[k] = []

            for k,v in vals.items():
                if isinstance(v, dict):
                    for kk,vv in v.items():
                        ret[k][kk].append(vv)
                else:
                    ret[k].append(v)

    return convert_array(ret)

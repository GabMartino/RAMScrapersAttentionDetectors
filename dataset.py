import asyncio
import os
from time import sleep

import numpy as np
import pandas as pd
import pyshark
from tqdm import tqdm
import pickle


def extractPacketsFromFile(path, N, subset=None):
    packetList = []
    count = 0
    #N = float("inf")
    packets = pyshark.FileCapture(path, keep_packets=False)
    for packet in tqdm(packets):
        # print(packet.udp.payload)
        preprocessedPacket = preprocessPacket(packet)  ## We take only TCP UDP packet, other packets are ("None")
        packetList.append(preprocessedPacket)
        count += 1
        if count >= N:
            break
    return packetList
'''
    Extract from the packets only necessary information
'''
def preprocessPacket(packet):
    try:
        if packet.ip.proto == "6":## TCP
            return (packet.ip.src,
                    packet.ip.dst,
                    packet.tcp.srcport, packet.tcp.dstport,
                    packet.tcp.flags_res,
                    packet.tcp.flags_ns,
                    packet.tcp.flags_cwr,
                    packet.tcp.flags_urg,
                    packet.tcp.flags_ack,
                    packet.tcp.flags_push,
                    packet.tcp.flags_reset,
                    packet.tcp.flags_syn,
                    packet.tcp.window_size_value,
                    packet.tcp.time_delta,
                    packet.ip.len
                    )


        elif packet.ip.proto == "17" : ##UDP
            return (packet.ip.src,
                    packet.ip.dst,
                    packet.udp.srcport, packet.udp.dstport,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0,
                    packet.udp.time_delta,
                    packet.ip.len
                    )
        else:
            return ("None")
    except:
        return ("None")
'''
    Extract Sessions from the list of packets
'''
def findFlows(listOfPackets):
    columns = ['SourceIP', 'DestIP','SourcePort', 'DestPort',
                                              'FlagRES', 'FlagNS', 'FlagCWR',
                                              'FlagURG', 'FlagACK', 'FlagPUSH',
                                              'FlagRESET', 'FlagSYN',
                                              'TCPWindowSize',
                                              'IAT',
                                              'Len']
    totalDatasetOfPackets = pd.DataFrame(listOfPackets, columns=columns)

    '''
        A flow is a series of packets that have the same set of "sourceIP"-"DestIP"
        A bi-flow/Session is a series of packets that have the same set of "sourceIP"-"DestIP" but also inverted
    '''

    BiFlows = {}
    for index, row in totalDatasetOfPackets.iterrows():
        SourceIP = row["SourceIP"]
        DestIP = row["DestIP"]
        key = SourceIP+"-"+DestIP
        inverseKey = DestIP+"-"+SourceIP
        if key in BiFlows:
            BiFlows[key].loc[len(BiFlows[key].index)] = row
        elif key not in BiFlows:
            if inverseKey in BiFlows:
                BiFlows[inverseKey].loc[len(BiFlows[inverseKey].index)] = row
            else:
                BiFlows[key] = pd.DataFrame(columns=columns)

    BiFlows = list(BiFlows.values())
    return BiFlows

'''
    1) Remove IP address
    2) Subdivide large flows in smaller flows 
        2.1) if flows are smaller than 5 packets -> Delete
        2.2) if flows are below N packets -> padd with zeros

'''

def preprocessBiFlows(Flows, NsizeFlows = 20):
    def split_given_size(a, size):
        return np.split(a, np.arange(size, len(a), size))

    Dataset = []
    for flow in tqdm(Flows):
        ##REMOVE COLUMNS

        flow = flow.drop('SourceIP', axis=1)
        flow = flow.drop('DestIP', axis=1)

        np_flow = flow.to_numpy()
        ## Create subflows of 20packets
        subFlows = split_given_size(np_flow, NsizeFlows)

        for subflow in subFlows:

            if subflow.shape[0] < NsizeFlows and subflow.shape[0] >= 2:
                # subflow = np.concatenate((subflow, np.zeros((NsizeFlows - subflow.shape[0], 0))), axis=1)
                subflow = np.pad(subflow, [(0, NsizeFlows - subflow.shape[0]), (0, 0)], mode='constant',
                                 constant_values=0)
                # np.concatenate((subflow, np.zeros((subflow.shape[0], NsizeFlows))), axis=1)
            if subflow.shape[0] < 2:
                break
            Dataset.append(subflow)

    return Dataset



def normalize(Flows):
    for index, flow in enumerate(Flows):
        try:
            flow = np.array(flow, dtype="float32")

            Flows[index][:, 0] = flow[:, 0] / 65535  ## port normalization
            Flows[index][:, 1] = flow[:, 1] / 65535  ##port normalization

            Flows[index][:, 10] = flow[:, 10] / 65535  ##  maximum tcp windwos
            Flows[index][:, 11] = flow[:, 11] / 65535
            Flows[index][:, 12] = flow[:, 12] / 65535  ## ip len packet max

        except:
            pass

    return Flows
if __name__ == "__main__":

    '''
        For each partial dataset take only the benign packets,
        assessed in "_labels.csv"
        
        1) Filter Benign packets
        2) Extract from every packets only necessary information
        3) Extract biflows
    
    '''
    TimeSeriesSize = 5
    basePathsource = "TransactionTracks/"
    basePath = "Datasets/POSTransaction/"
    dir = "POSmalwareTracks"
    from glob import glob

    paths = glob(dir+"/*/*", recursive=True)
    MalignPackets = [path for path in paths if "MalignPackets" in path]

    pathPcap = basePath + "CaputreSep29_1213_1246.pcapng"
    #paths = [basePathsource + "CaputreSep29_1035_1113.pcapng", basePathsource + "CaputreSep29_1115_1211.pcapng", basePathsource + "CaputreSep29_1213_1246.pcapng"]

    paths = ["POSmalwareTracks/All/https_only.pcapng"]
    #paths = MalignPackets
    packetList = []
    N = float("inf")
    for path in paths:
        packets = extractPacketsFromFile(path, N)
        packetList.append(packets)

    packetList = [item for sublist in packetList for item in sublist]
    print("Total number of packets: ", len(packetList))

    BenignPackets = [ p for p in packetList if p != ("None")] ##filter out non valid packets
    
    BenignFlows = findFlows(BenignPackets)
    print("Total number of biflows extracted: ", len(BenignFlows))
    print("Number of packets after biflows extraction: ", sum([len(f.index) for f in BenignFlows]))
    PreprocessedBiflows = preprocessBiFlows(BenignFlows, NsizeFlows=TimeSeriesSize)

    NormalizedBiflows = normalize(PreprocessedBiflows)
    print("Number of biflows after splitting for N packets per flows: ", len(NormalizedBiflows))
    outPath = "https_only_malign" #"BenignBiFlows"
    with open(basePath + str(TimeSeriesSize) + "/" + outPath, 'wb') as f:
        pickle.dump(NormalizedBiflows, f)


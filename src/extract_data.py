import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xml.etree.ElementTree as ET
import struct


def extract_QS_from_XML(prefix_per_patient):
    file_path = prefix_per_patient + "_OCTAnalysisInfo.xml"
    tree = ET.parse(file_path)
    root = tree.getroot()

    tag = "RetinalImageQuality"
    data_for_QS = []

    for elem in root.findall(f'.//{tag}'):
        data_for_QS.append(elem.text)

    QS = float(data_for_QS[0])
    return QS

# 使用例
# prefix_per_patient = "data_xml/00120141010_20141010_R_71"
# extract_QS_from_XML(prefix_per_patient)


def read_boundary(file_path):
    # ファイルをバイナリモードで開く
    with open(file_path, 'rb') as file:
        data = file.read()

    # バイナリデータを16ビット整数のリストに変換
    # '<' はリトルエンディアン、'H' はunsigned short (16ビット整数) 
    data_int = struct.unpack('<' + 'H' * (len(data) // 2), data)
    data_int = np.array(data_int).reshape(128, 512).T

    return data_int

def extract_layers_from_DAT(prefix_per_patient):
    prefix_per_patient += "_BOUNDARY"
    BDY_ILM = read_boundary(prefix_per_patient+'_ILM.dat')
    # RNFL
    BDY_RNFLGCL = read_boundary(prefix_per_patient+'_RNFLGCL.dat')
    # GCL
    BDY_IPLINL = read_boundary(prefix_per_patient+'_IPLINL.dat')
    # OUTER
    # BDY_ISOS = read_boundary(prefix_per_patient+'_ISOS.dat')
    # BDY_OSRPE = read_boundary(prefix_per_patient+'_OSRPE.dat')
    BDY_BM = read_boundary(prefix_per_patient+'_BM.dat')

    # 差分
    RNFL = BDY_ILM - BDY_RNFLGCL
    GCL = BDY_RNFLGCL - BDY_IPLINL
    OUTER = BDY_IPLINL - BDY_BM

    # 右目の向きに統一。なぜか右目の場合のみ反転が必要
    if "_R_" in prefix_per_patient:
        RNFL = RNFL[:,::-1]
        GCL = GCL[:,::-1]
        OUTER = OUTER[:,::-1]

    return RNFL, GCL, OUTER

# 使用例
# prefix_per_patient = "data_dat/00120141010_20141010_R_71_BOUNDARY"
# RNFL, GCL, OUTER = extract_layers_from_DAT(prefix_per_patient)



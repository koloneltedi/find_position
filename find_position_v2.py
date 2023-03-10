# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:34:41 2023

@author: aivlev
"""

#### NOTE: THE DATA IS STORED IN A LIST-BASED WAY. NOT REALLY A BOTTLENECK OF SPEED, BUT IT IS UGLY. FOR A VERSION 2, REMAKE IT SUCH THAT IT IS SAVED AS A NUMPY ARRAY (WITHIN DICTIONARIES)

import os

from os import listdir
from os.path import isfile, join
import pyaedt
import numpy as np
from matplotlib import pyplot as plt
import scipy
import time
import matplotlib as mpl
import matplotlib.image as mpimg

import pickle

from collections import OrderedDict


def open_q3d():
    non_graphical = os.getenv("PYAEDT_NON_GRAPHICAL", "False").lower() in ("true", "1", "t")
    
    q = pyaedt.Q3d(projectname=pyaedt.generate_unique_project_name(),
                   specified_version="2022.2",
                   non_graphical=non_graphical,
                   new_desktop_session=True)
    
    return q
    
def close_q3d(q):
    q.release_desktop(close_projects=True, close_desktop=True)
    

def define_gates_4dot():
    objects_dict = {}
    
    
    SL_dict = {}
    SL_dict['height'] = 40*1e-3 ## in um
    SL_plunger_zpos= 12*1e-3 #in um
    SL_lead_zpos = 32*1e-3 #in um
    SL_dict['geometry']  = np.array([[-0.222,-0.28,SL_plunger_zpos],[-.264,-.238,SL_plunger_zpos],[-.264,-.182,SL_plunger_zpos],[-.222,-.14,SL_plunger_zpos],[-.166,-.14,SL_plunger_zpos],[-.124,-.182,SL_plunger_zpos],[-.124,-.238,SL_plunger_zpos],[-.166,-.28,SL_plunger_zpos],[-.222,-.28,SL_plunger_zpos]])
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    SL_dict['SLgeometry_1']  = np.array([[(-17.086*mult_factor1+-.242*mult_factor2),(-11.1*mult_factor1+-.254*mult_factor2),SL_lead_zpos],[(-17.076*mult_factor1+-.263*mult_factor2),(-8.88*mult_factor1+-.213*mult_factor2),SL_lead_zpos],[-.263,-.213,SL_lead_zpos],[-.242,-.254,SL_lead_zpos],[(-17.086*mult_factor1+-.242*mult_factor2),(-11.1*mult_factor1+-.254*mult_factor2),SL_lead_zpos]])
    #geometry in um
    SL_dict["merge_list"] = ["geometry","SLgeometry_1"]
    SL_dict['material'] = "palladium"
    objects_dict["SL"] = SL_dict
    
    SSL_dict = {}
    SSL_dict['height'] = 20*1e-3 ## in um
    SSL_zpos = 7e-3 #um
    SSL_dict['geometry']  = np.array([[-0.560,-0.264,SSL_zpos],[-0.402,-0.263,SSL_zpos],[-0.282,-0.208,SSL_zpos],[-0.252,-0.268,SSL_zpos],[-0.434,-0.383,SSL_zpos],[-0.476,-0.301,SSL_zpos],[-0.560,-0.264,SSL_zpos]])
    #geometry in um
    SSL_dict["merge_list"] = ["geometry"]
    SSL_dict['material'] = "palladium"
    objects_dict["SSL"] = SSL_dict
    
    BLU_dict = {}
    BLU_dict['height'] = 20*1e-3 ## in um
    BLU_zpos = 7e-3 #um
    BLU_dict['geometry']  = np.array([[-0.337,-0.178,BLU_zpos],[-0.33700,-0.14100,BLU_zpos],[-0.28000,-0.14100,BLU_zpos],[-0.23400,-0.10000,BLU_zpos],[-0.22000,-0.10000,BLU_zpos],[-0.21200,-0.10800,BLU_zpos],[-0.21200,-0.12200,BLU_zpos],[-0.26800,-0.17800,BLU_zpos],[-0.337, -0.17800,BLU_zpos]])
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    BLU_dict['BLUgeometry_1']  = np.array([[-12.54400*mult_factor1+-0.37600*mult_factor2,0.83000*mult_factor1+-0.14100*mult_factor2,BLU_zpos],[-0.37600, -0.14100,BLU_zpos],[-0.3330,-0.14100,BLU_zpos],[-0.33300,-0.17800,BLU_zpos],[-0.36800,-0.17800,BLU_zpos],[-12.5600*mult_factor1+-0.36800*mult_factor2,-0.69300*mult_factor1+-0.17800*mult_factor2,BLU_zpos],[-12.54400*mult_factor1+-0.37600*mult_factor2,0.83000*mult_factor1+-0.14100*mult_factor2,BLU_zpos]])
    #geometry in um
    BLU_dict["merge_list"] = ["geometry","BLUgeometry_1"]
    BLU_dict['material'] = "palladium"
    objects_dict["BLU"] = BLU_dict
    
    BLD_dict = {}
    BLD_dict['height'] = 20*1e-3 ## in um
    BLD_zpos = 7e-3 #um
    BLD_dict['geometry']  = np.array([[-0.22800	,-0.34200,BLD_zpos],[-0.25800,-0.32000,BLD_zpos],[-0.25400,	-0.31600,BLD_zpos],[-0.22500,	-0.28500,BLD_zpos],[-0.14700,	-0.28500,BLD_zpos],[-0.13900,	-0.29300,BLD_zpos],[-0.13900,	-0.30600,BLD_zpos],[-0.14800,	-0.31500,BLD_zpos],[-0.20900,	-0.31500,BLD_zpos],[-0.22800,	-0.34200,BLD_zpos]])
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    BLD_dict['BLDgeometry_1']  = np.array([[-17.00000*mult_factor1-0.247*mult_factor2,-15.53000*mult_factor1+-0.36600*mult_factor2,BLD_zpos],[-17.0860*mult_factor1+-.287*mult_factor2,-13.100*mult_factor1-0.3490*mult_factor2,BLD_zpos],[-0.28700,	-0.34900,BLD_zpos],[-0.25500,	-0.31700,BLD_zpos],[-0.22600	,-0.33900,BLD_zpos],[-0.22800,	-0.34200,BLD_zpos],[-0.24700,	-0.36600,BLD_zpos],[-17.00000*mult_factor1-0.247*mult_factor2,-15.53000*mult_factor1+-0.36600*mult_factor2,BLD_zpos]])
    #geometry in um
    BLD_dict["merge_list"] = ["geometry","BLDgeometry_1"]
    BLD_dict['material'] = "palladium"
    objects_dict["BLD"] = BLD_dict
    
    BL_dict = {}
    BL_dict['height'] = 20*1e-3 ## in um
    BL_zpos = 7e-3 #um
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    BL_dict['geometry']  = np.array([[0.72500*mult_factor1+-0.044*mult_factor2,	-17.03400*mult_factor1+-0.354*mult_factor2,BL_zpos],[-1.08100*mult_factor1+-0.10100*mult_factor2,	-17.01000*mult_factor1+-0.36000*mult_factor2,BL_zpos],[-0.10100,	-0.36000,BL_zpos],[-0.12200,	-0.18400,BL_zpos],[-0.17800,	-0.12800,BL_zpos],[-0.17800,	-0.10700,BL_zpos],[-0.16000,	-0.08600,BL_zpos],[-0.13800,	-0.08700,BL_zpos],[-0.07800,	-0.14900,BL_zpos],[-0.07700,	-0.15200,BL_zpos],[-0.07700,	-0.15600,BL_zpos],[-0.04400,	-0.35400,BL_zpos],[0.72500*mult_factor1+-0.044*mult_factor2,	-17.03400*mult_factor1+-0.354*mult_factor2,BL_zpos]])
    BL_dict['geometry'][:,0] += 0.005
    #geometry in um
    BL_dict["merge_list"] = ["geometry"]
    BL_dict['material'] = "palladium"
    objects_dict["BL"] = BL_dict
    
    
    return objects_dict

def define_gates():
    objects_dict = {}
    
    SL_dict = {}
    SL_dict['height'] = 40*1e-3 ## in um
    SL_plunger_zpos= 12*1e-3 #in um
    SL_lead_zpos = 32*1e-3 #in um
    SL_dict['geometry']  = np.array([[-0.28200,	-0.08400,SL_plunger_zpos],[-0.32400,	-0.04200,SL_plunger_zpos],[-0.32400,	0.01400,SL_plunger_zpos],[-0.28200,	0.05600,SL_plunger_zpos],[-0.22600,	0.05600,SL_plunger_zpos],[-0.18400,	0.01400,SL_plunger_zpos],[-0.18400,	-0.04200,SL_plunger_zpos],[-0.22600,	-0.08400,SL_plunger_zpos],[-0.28200,	-0.08400,SL_plunger_zpos]])
    SL_dict['SLgeometry_1']  = np.array([[-0.5100,	0.01100,SL_lead_zpos],[-0.33500,	0.01100,SL_lead_zpos],[-0.33500,	-0.03900,SL_lead_zpos],[-0.5100,	-0.04000,SL_lead_zpos],[-0.5100,	0.01100,SL_lead_zpos]])
    SL_dict['SLgeometry_2']  = np.array([[-0.33900,	0.01100,SL_plunger_zpos],[-0.32200,	0.01100,SL_plunger_zpos],[-0.32200,	-0.03900,SL_plunger_zpos],[-0.33900,	-0.04000,SL_plunger_zpos],[-0.33900,	0.01100,SL_plunger_zpos]])
    SL_dict['SLgeometry_3']  = np.array([[-0.76200,	0.01100,SL_plunger_zpos],[-0.50700,	0.01100,SL_plunger_zpos],[-0.50700,	-0.03900,SL_plunger_zpos],[-0.76200,	-0.04000,SL_plunger_zpos],[-0.76200,	0.01100,SL_plunger_zpos]])
    #geometry in um
    SL_dict["merge_list"] = ["geometry","SLgeometry_1","SLgeometry_2","SLgeometry_3"]
    SL_dict['material'] = "palladium"
    objects_dict["SL"] = SL_dict
    
    SSL_dict = {}
    SSL_dict['height'] = 20*1e-3 ## in um
    SSL_zpos = 7e-3 #um
    mult_factor1 = 0.02
    mult_factor2 = 1-mult_factor1
    SSL_dict['geometry']  = np.array([[-0.50200,	-0.05000,SSL_zpos],[-0.50200,	0.03300,SSL_zpos],[-0.60200,	0.04800,SSL_zpos],[-12.80800*mult_factor1+-0.60200*mult_factor2,	1.74000*mult_factor1+0.04800*mult_factor2,SSL_zpos],[-12.91900*mult_factor1+-0.60300*mult_factor2,	2.75000*mult_factor1+0.09600*mult_factor2,SSL_zpos],[-0.60300,	0.09600,SSL_zpos],[-0.53100,	0.07600,SSL_zpos],[-0.34400,	0.02500,SSL_zpos],[-0.34400,	-0.05000,SSL_zpos],[-0.50200,	-0.05000,SSL_zpos]])
    #geometry in um
    SSL_dict["merge_list"] = ["geometry"]
    SSL_dict['material'] = "palladium"
    objects_dict["SSL"] = SSL_dict
    
    BLU_dict = {}
    BLU_dict['height'] = 20*1e-3 ## in um
    BLU_zpos = 7e-3 #um
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    BLU_dict['geometry']  = np.array([[-0.28900,	0.06000,BLU_zpos],[-17.22400*mult_factor1+-0.28900*mult_factor2,	5.34000*mult_factor1+0.06000*mult_factor2,BLU_zpos],[-17.22400*mult_factor1+-0.28600*mult_factor2,	7.34000*mult_factor1+0.09100*mult_factor2,BLU_zpos],[-0.28600,	0.09100,BLU_zpos],[-0.22500,	0.09100,BLU_zpos],[-0.21400,	0.08000,BLU_zpos],[-0.21400,	0.07000,BLU_zpos],[-0.22400,	0.06000,BLU_zpos],[-0.28900,	0.06000,BLU_zpos]])
    #geometry in um
    BLU_dict["merge_list"] = ["geometry"]
    BLU_dict['material'] = "palladium"
    objects_dict["BLU"] = BLU_dict
    
    BLD_dict = {}
    BLD_dict['height'] = 20*1e-3 ## in um
    BLD_zpos = 7e-3 #um
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    BLD_dict['geometry']  = np.array([[-0.28900,	-0.08800,BLD_zpos],[-17.22400*mult_factor1+-0.28900*mult_factor2,	-5.36800*mult_factor1+-0.08800*mult_factor2,BLD_zpos],[-17.22400*mult_factor1+-0.28600*mult_factor2,	-7.36800*mult_factor1+-0.11900*mult_factor2,BLD_zpos],[-0.28600,	-0.11900,BLD_zpos],[-0.22500,	-0.11900,BLD_zpos],[-0.21400,	-0.10800,BLD_zpos],[-0.21400,	-0.09800,BLD_zpos],[-0.22400,	-0.08800,BLD_zpos],[-0.28900,	-0.08800,BLD_zpos]])
    #geometry in um
    BLD_dict["merge_list"] = ["geometry"]
    BLD_dict['material'] = "palladium"
    objects_dict["BLD"] = BLD_dict
    
    BL_dict = {}
    BL_dict['height'] = 20*1e-3 ## in um
    BL_zpos = 7e-3 #um
    BL_dict['geometry']  = np.array([[-0.18400,	-0.09000,BL_zpos],[-0.18400,	0.03500,BL_zpos],[-0.16800,	0.05100,BL_zpos],[-0.15000,	0.05100,BL_zpos],[-0.13400,	0.03500,BL_zpos],[-0.13400,	-0.09000,BL_zpos],[-0.18400,	-0.09000,BL_zpos]])
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    BL_dict['BLgeometry_1']  = np.array([[-0.13500,	-0.25000,BL_zpos],[-2.65400*mult_factor1+-0.13500*mult_factor2,	-12.05700*mult_factor1+-0.25000*mult_factor2,BL_zpos],[-3.65600*mult_factor1+-0.19000*mult_factor2,	-12.04500*mult_factor1+-0.25000*mult_factor2,BL_zpos],[-0.19000,	-0.25000,BL_zpos],[-0.18400,	-0.16000,BL_zpos],[-0.18400,	-0.08500,BL_zpos],[-0.13400,	-0.08500,BL_zpos],[-0.13500,	-0.25000,BL_zpos]])
    
    BL_dict['geometry'][:,0] += 0.005
    BL_dict['BLgeometry_1'][:,0] += 0.005
    #geometry in um
    BL_dict["merge_list"] = ["geometry",'BLgeometry_1']
    BL_dict['material'] = "palladium"
    objects_dict["BL"] = BL_dict
    
    OLD_dict = {}
    OLD_dict['height'] = 20*1e-3 ## in um
    OLD_zpos = 7e-3 #um
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    OLD_dict['geometry']  = np.array([[-0.385,	-0.233,OLD_zpos],[-17.226*mult_factor1+-0.385*mult_factor2,	-10.572*mult_factor1+-0.233*mult_factor2,OLD_zpos],[-17.226*mult_factor1+-0.336*mult_factor2,	-12.63*mult_factor1+-0.272*mult_factor2,OLD_zpos],[-0.336,	-0.272,OLD_zpos],[-0.232,	-0.126,OLD_zpos],[-0.288,	-0.126,OLD_zpos],[-0.385,	-0.233,OLD_zpos]])
    
    #geometry in um
    OLD_dict["merge_list"] = ["geometry"]
    OLD_dict['material'] = "palladium"
    objects_dict["OLD"] = OLD_dict
    
    OLU_dict = {}
    OLU_dict['height'] = 20*1e-3 ## in um
    OLU_zpos = 7e-3 #um
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    OLU_dict['geometry']  = np.array([[-0.384,	0.205,OLU_zpos],[-17.225*mult_factor1+-0.384*mult_factor2,	10.544*mult_factor1+0.205*mult_factor2,OLU_zpos],[-17.225*mult_factor1+-0.335*mult_factor2,	12.602*mult_factor1+0.244*mult_factor2,OLU_zpos],[-0.335,	0.244,OLU_zpos],[-0.231,	0.098,OLU_zpos],[-0.287,	0.098,OLU_zpos],[-0.384,	0.205,OLU_zpos]])
    
    #geometry in um
    OLU_dict["merge_list"] = ["geometry"]
    OLU_dict['material'] = "palladium"
    objects_dict["OLU"] = OLU_dict
    
    OBD_dict = {}
    OBD_dict['height'] = 20*1e-3 ## in um
    OBD_zpos = 7e-3 #um
    mult_factor1 = 0.03
    mult_factor2 = 1-mult_factor1
    OBD_dict['geometry']  = np.array([[-0.207,-0.25,OLD_zpos],[-6.597*mult_factor1+-0.207*mult_factor2,	-11.7*mult_factor1+-0.25*mult_factor2,BL_zpos],[-7.597*mult_factor1+-0.257*mult_factor2,	-11.7*mult_factor1+-0.25*mult_factor2,BL_zpos],[-0.257,	-0.25,BL_zpos],[-0.212,	-0.125,BL_zpos],[-0.211,	-0.122,BL_zpos],[-0.19,	-0.122,BL_zpos],[-0.19,	-0.125,BL_zpos],[-0.207,	-0.250,BL_zpos]])
    
    #geometry in um
    OBD_dict["merge_list"] = ["geometry"]
    OBD_dict['material'] = "palladium"
    objects_dict["OBD"] = OBD_dict
    
    P1_dict = {}
    P1_dict['height'] = 40*1e-3 ## in um
    P1_zpos = 12e-3 #um
    P1_dict['geometry']  = np.array([[-0.06000,	-0.07300,P1_zpos],[-0.02600,	-0.03900,P1_zpos],[-0.02600,	0.00100,P1_zpos],[-0.06000,	0.03500,P1_zpos],[-0.10000,	0.03500,P1_zpos],[-0.13400,	0.00100,P1_zpos],[-0.13400,	-0.03900,P1_zpos],[-0.10000,	-0.07300,P1_zpos],[-0.06000,	-0.07300,P1_zpos]])
    P1_dict['geometry'][:,0] += 0.010
    
    
    #geometry in um
    P1_dict["merge_list"] = ["geometry"]
    P1_dict['material'] = "palladium"
    objects_dict["P1"] = P1_dict
    
    
    return objects_dict

def make_gates(q,objects_dict):
    
    all_gate_names = list(objects_dict.keys())
    
    for key in all_gate_names:
        merge_list = objects_dict[key]["merge_list"]
        for geometry_name in merge_list:
            print(geometry_name)
            q.modeler.create_polyline(objects_dict[key][geometry_name],name=geometry_name)
            q.modeler.cover_lines(geometry_name)
            q.modeler.sweep_along_vector(geometry_name,[0,0,objects_dict[key]["height"]])
        q.modeler.unite(merge_list)
        q.modeler[merge_list[0]].name = key
        
        q.assign_material(key,objects_dict[key]["material"])
        
    q.modeler.create_box([-1, -1, 0], [1.5, 1.7, 0.012], name="ALD_1", matname="Al2_O3_ceramic")
    q.modeler["ALD_1"].color = (100, 100, 100)
    q.modeler["ALD_1"].transparency = 0.8
    
    all_gate_names.remove('SL')
    all_gate_names.remove('P1')
    for gate in all_gate_names:
        ALD_extra_name = "ALD_2_"+gate
        q.modeler.create_box([-1, -1, 0], [1.5, 1.7, 0.012], name=ALD_extra_name, matname="Al2_O3_ceramic")
        q.modeler[ALD_extra_name].color = (100, 100, 100)
        q.modeler[ALD_extra_name].transparency = 0.8
        q.modeler.intersect([ALD_extra_name,gate], keep_originals=True)
        q.modeler.move(ALD_extra_name,[0,0,0.02])
    
    
    q.modeler.subtract('ALD_1',all_gate_names)

def make_substrate(q):
    substrate_pos_xy = (-1, -1)
    substrate_size_xy = (1.5,1.7)
    
    
    q.modeler.create_box([substrate_pos_xy[0], substrate_pos_xy[1], -0.055], [substrate_size_xy[0], substrate_size_xy[1], 0.055], name="SiGe1", matname="Si2Ge8")
    q.modeler.create_box([substrate_pos_xy[0], substrate_pos_xy[1], -0.065], [substrate_size_xy[0], substrate_size_xy[1], 0.010], name="Ge1", matname="germanium")
    q.modeler.create_box([substrate_pos_xy[0], substrate_pos_xy[1], -0.069], [substrate_size_xy[0], substrate_size_xy[1], 0.004], name="SiGe2", matname="Si2Ge8")
    q.modeler.create_box([substrate_pos_xy[0], substrate_pos_xy[1], -0.085], [substrate_size_xy[0], substrate_size_xy[1], 0.016], name="Ge2", matname="germanium")
    q.modeler.create_box([substrate_pos_xy[0], substrate_pos_xy[1], -0.160], [substrate_size_xy[0], substrate_size_xy[1], 0.075], name="SiGe3", matname="Si2Ge8")
    
    heterstructure_colors = {"SiGe1":(230,60,120),"SiGe2":(230,60,120),"SiGe3":(230,60,120),"Ge2":(250,80,80)}
    
    
    for key in heterstructure_colors.keys():
        q.modeler[key].color = heterstructure_colors[key]
        q.modeler[key].transparency = 0.8

def make_dot(q,position,radius=0.02,layer = 'top'):
    if layer == 'top':
        z_pos = -0.065
        thickness = 0.01
        Ge_name = 'Ge1'
    elif layer == 'bot':
        z_pos = -0.085
        thickness = 0.016
        Ge_name = 'Ge2'
    
    position = [position[0],position[1],z_pos]
    
    dot_object = q.modeler.create_circle(2,position,radius,name="dot")
    q.modeler.sweep_along_vector(dot_object.name,[0,0,thickness])
    q.assign_material(dot_object.name,"pec")    
    
    q.modeler.subtract([Ge_name],['dot'])

def analyse(q,SaveFields=True):
    q.auto_identify_nets()
    # print(q.nets)

    Cap_props = {"MaxPass":15}
    setup1 = q.create_setup(setupname = 'MySetup' , props={'Cap':Cap_props,"AdaptiveFreq": "1MHz", "SaveFields": SaveFields, "DC": False, "AC": False})
    
    data_plot_mutual = q.get_traces_for_plot(get_self_terms=False, get_mutual_terms=True, category="C")
    # q.post.create_report(expressions=data_plot_mutual, context="Original", plot_type="Data Table")
    
    q.analyze_nominal()
    
    result = q.post.get_solution_data(expressions=data_plot_mutual, context="Original")
               
    return result

def clear_analysis(q):
    
    q.delete_setup('MySetup')
    delete_list = ['dot','SiGe1','SiGe2','SiGe3','Ge1','Ge2']
    q.modeler.delete(delete_list)

def reshuffle_list(gate_list,rel_cap_list,desired_order = ['BL','SSL','BLU','BLD']):
    new_cap_list = [None] * len(desired_order)
    
    for idx,gate in enumerate(desired_order):
        index = gate_list.index(gate)
        new_cap_list[idx] = rel_cap_list[index]
               
    return desired_order,new_cap_list

def post_process_result(result):
    dot_capacitance = {}

    for key in result._solutions_real.keys():
        if "dot" in key.split(',')[0]:
            subkey = list(result._solutions_real[key].keys())[-1]
            cap_value = result._solutions_real[key][subkey]
            
            unit = result.units_data[key]
            if unit == 'pF':
                factor = 1e-12
            elif unit == 'fF':
                factor = 1e-15
            else:
                print("UNKNOWN UNIT: "+unit)
                
            dot_capacitance[key.split(',')[1][:-1]] = cap_value*factor

    main_gate = "SL"

    gate_list = []
    rel_cap_list = []
    abs_cap_list = []

    for key in dot_capacitance.keys():
        relative_capacitance = dot_capacitance[key]/dot_capacitance[main_gate]
        
        rel_cap_list.append(relative_capacitance)
        abs_cap_list.append(dot_capacitance[key])
        gate_list.append(key)        
        print(key+": "+str(relative_capacitance))
    
    return gate_list,rel_cap_list,abs_cap_list
        
def calculate_cost_for_rad(results_rad_dict,goal_rel_cap = {'BL':0.45,'SSL':0.5,'BLU':0.4,'BLD':0.4}):
    cost = 0
    
    main_gate = 'SL'
    for gate in goal_rel_cap.keys():
        cost += (results_rad_dict[gate]/results_rad_dict[main_gate]-goal_rel_cap[gate])**2
            
    return cost

def plot_results(gate_list,rel_cap_list):
    plt.figure()
    plt.plot(range(len(gate_list)),rel_cap_list,'o-')    
    plt.xticks(range(len(gate_list)),gate_list)
    plt.ylim(bottom=0)
    
def run_sequence(q):
    N_x,N_y,N_r = 10,10,3
    pos_x_list = np.linspace(-0.35,-0.18,N_x)
    pos_y_list = np.linspace(-0.09,0.06,N_y)
    radius_list = np.linspace(0.03,0.07,N_r)
    
    rad_dict = {}
    
    for idx_rad, rad in enumerate(radius_list):
        
        X_pos = np.empty((N_x,N_y))
        Y_pos = np.empty(np.shape(X_pos))
        
        SL_array = np.empty(np.shape(X_pos))
        BL_array = np.empty(np.shape(X_pos))
        SSL_array = np.empty(np.shape(X_pos))
        BLU_array = np.empty(np.shape(X_pos))
        BLD_array = np.empty(np.shape(X_pos))
        
        for idx_x, pos_x in enumerate(pos_x_list):
            for idx_y, pos_y in enumerate(pos_y_list):    
                                         
                X_pos[idx_x,idx_y] = pos_x
                Y_pos[idx_x,idx_y] = pos_y
                
                try:
                    make_substrate(q)
                    make_dot(q,[pos_x,pos_y],rad,layer='top')
                    result = analyse(q,SaveFields=False)
                    clear_analysis(q)
                    gate_list,rel_cap_list,abs_cap_list = post_process_result(result)
                except:
                    input("Didn't work. Maybe connection to licence server lost. Try again! \n[Press Any Button]")
                    make_substrate(q)
                    make_dot(q,[pos_x,pos_y],rad,layer='top')
                    result = analyse(q,SaveFields=False)
                    clear_analysis(q)
                    gate_list,rel_cap_list,abs_cap_list = post_process_result(result)
                
                SL_array[idx_x,idx_y] = abs_cap_list[gate_list.index('SL')]
                BL_array[idx_x,idx_y] = abs_cap_list[gate_list.index('BL')]
                SSL_array[idx_x,idx_y] = abs_cap_list[gate_list.index('SSL')]
                BLU_array[idx_x,idx_y] = abs_cap_list[gate_list.index('BLU')]
                BLD_array[idx_x,idx_y] = abs_cap_list[gate_list.index('BLD')]
                
                
        rad_dict['SL'] = SL_array
        rad_dict['BL'] = BL_array
        rad_dict['SSL'] = SSL_array
        rad_dict['BLU'] = BLU_array
        rad_dict['BLD'] = BLD_array
        rad_dict['X'] = X_pos
        rad_dict['Y'] = Y_pos
        rad_dict['rad'] = rad
        
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = current_dir+"\\FEM_results_2"
        name = f"dotRadius_{rad}_relCapList_{str(int(time.time()))}.pkl"
        
        os.chdir(target_dir)

        with open(name, 'wb') as f:
            pickle.dump(rad_dict, f)

    return rad_dict

def load_results():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = current_dir+"\\FEM_results_2"
    os.chdir(target_dir)
    
    only_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    
    results_dict = {}
    
    for idx,file in enumerate(only_files):
        rad = float(file.split('_')[1])
        rad = round(rad,3) #um
        
        with open(file, "rb") as input_file:
            results_dict[str(rad)] = pickle.load(input_file)
    
    return results_dict


def interpolate_results(results_dict,N_x=100,N_y=100):
    from scipy.interpolate import RegularGridInterpolator as interpolator
    
    for key in results_dict.keys():
        old_X = results_dict[key]['X']
        old_Y = results_dict[key]['Y']
        
        min_x,max_x = old_X[0,0],old_X[-1,0]
        min_y,max_y = old_Y[0,0],old_Y[0,-1]
        
        new_x = np.linspace(min_x,max_x,N_x)
        new_y = np.linspace(min_y,max_y,N_y)
        
        new_X,new_Y = np.meshgrid(new_x,new_y,indexing = 'ij')
        
        results_dict[key]['X'] = new_X
        results_dict[key]['Y'] = new_Y
        
        new_X = np.reshape(new_X,np.shape(new_X)+(1,))
        new_Y = np.reshape(new_Y,np.shape(new_Y)+(1,))
        points = np.concatenate((new_X,new_Y),axis=2)
        
        method = 'cubic'
        results_dict[key]['SL'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[key]['SL'],method=method)(points)
        results_dict[key]['BL'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[key]['BL'],method=method)(points)
        results_dict[key]['SSL'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[key]['SSL'],method=method)(points)
        results_dict[key]['BLU'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[key]['BLU'],method=method)(points)
        results_dict[key]['BLD'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[key]['BLD'],method=method)(points)
    
    return results_dict

def plot_layout(plot_name="Expected position",image_name="design_screenshot.png"):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)
    
    
    img = mpimg.imread(image_name)
    
    plt.figure(plot_name)
    plt.imshow(img,extent=[-0.6,-0.113,-0.185,0.157])

def plot_results_dict(results_dict,max_cost_cut = 0.1,contours = True,plot_relcap = True):
    all_goals = {}
    
    all_goals["dot1"] = {'BL':0.45,'SSL':0.53,'BLU':0.41,'BLD':0.42} # dot1
    # all_goals["dot1 displaced"] = {'BL':0.45,'SSL':0.53,'BLU':0.55,'BLD':0.3} # dot1 displaced
    all_goals["dot2"] = {'BL':0.55,'SSL':0.73,'BLU':0.54,'BLD':0.61} # dot2
    
    all_goal_colors = {"dot1":'blue',"dot1 displaced":'green',"dot2":'orange'}
    
    plot_layout()
    
    for goal_key in all_goals.keys():
        
        goal_rel_cap = all_goals[goal_key]
    
        min_idx_list = []
        min_cost_list = []
        min_xy_list = []
        rad_list = []
        
        for key in results_dict.keys():
            radius = float(key)
            rad_list.append(1000*radius)
            
            X = results_dict[key]['X']
            Y = results_dict[key]['Y']
            
            SL_abs = results_dict[key]['SL']
            BL_rel = results_dict[key]['BL']/SL_abs
            SSL_rel = results_dict[key]['SSL']/SL_abs
            BLU_rel = results_dict[key]['BLU']/SL_abs
            BLD_rel = results_dict[key]['BLD']/SL_abs
            
            if plot_relcap:
                
                plt.figure()
                plt.title(f'BL rel_cap for {goal_key} radius {1000*radius}nm')
                im = plt.pcolor(X,Y,BL_rel)
                plt.colorbar(im,label='rel_cap')
                
                plt.figure()
                plt.title(f'SSL rel_cap for {goal_key} radius {1000*radius}nm')
                im = plt.pcolor(X,Y,SSL_rel)
                plt.colorbar(im,label='rel_cap')
                
                plt.figure()
                plt.title(f'BLU rel_cap for {goal_key} radius {1000*radius}nm')
                im = plt.pcolor(X,Y,BLU_rel)
                plt.colorbar(im,label='rel_cap')
                
                plt.figure()
                plt.title(f'BLD rel_cap for {goal_key} radius {1000*radius}nm')
                im = plt.pcolor(X,Y,BLD_rel)
                plt.colorbar(im,label='rel_cap')
                plt.show()
            
            
            epsilon = 0.03
            BL_close = np.abs(BL_rel-goal_rel_cap['BL'])<epsilon
            SSL_close = np.abs(SSL_rel-goal_rel_cap['SSL'])<epsilon
            BLU_close = np.abs(BLU_rel-goal_rel_cap['BLU'])<epsilon
            BLD_close = np.abs(BLD_rel-goal_rel_cap['BLD'])<epsilon
            
            
            plt.figure()
            plt.title(f'Cost for {goal_key} radius {1000*radius}nm')
            cost = calculate_cost_for_rad(results_dict[key],goal_rel_cap = goal_rel_cap)
            im = plt.pcolor(X,Y,cost,vmin=0, vmax=max_cost_cut)
            plt.colorbar(im,label='cost')
            
            if contours:
                plt.pcolor(X, Y, BL_close, shading='auto',alpha=0.5*BL_close,cmap=mpl.colormaps['Reds'])
                plt.pcolor(X, Y, SSL_close, shading='auto',alpha=0.5*SSL_close,cmap=mpl.colormaps['Purples'])
                plt.pcolor(X, Y, BLU_close, shading='auto',alpha=0.5*BLU_close,cmap=mpl.colormaps['Greens'])
                plt.pcolor(X, Y, BLD_close, shading='auto',alpha=0.5*BLD_close,cmap=mpl.colormaps['Blues'])
            
            # plt.show()
            
            min_idx = np.argmin(cost)
            min_cost = np.ndarray.flatten(cost)[min_idx]
            
            
            min_idx_list.append(min_idx)
            min_cost_list.append(min_cost)
            min_xy = [np.ndarray.flatten(X)[min_idx],np.ndarray.flatten(Y)[min_idx]]
            min_xy_list.append(min_xy)
            
            
        min_idx_arr = np.array(min_idx_list)
        min_cost_arr = np.array(min_cost_list)
        min_xy_arr = np.array(min_xy_list)
        rad_arr = np.array(rad_list)
        
        best_rad_idx = np.argmin(min_cost_arr)
        
        plt.figure("Expected Radius")
        plt.plot(rad_arr,min_cost_arr,color = all_goal_colors[goal_key],label=goal_key)
        plt.plot(rad_arr[best_rad_idx],min_cost_arr[best_rad_idx],'*',markersize=10,color = all_goal_colors[goal_key])
        plt.legend()
        # plt.show()
        
        plt.figure("Expected position")
        plt.plot(min_xy_arr[:,0],min_xy_arr[:,1],'o-',color = all_goal_colors[goal_key],label=goal_key)
        plt.plot(min_xy_arr[best_rad_idx,0],min_xy_arr[best_rad_idx,1],'*',markersize=12,color = all_goal_colors[goal_key])
        # plt.xlim([np.min(X),np.max(X)])
        # plt.ylim([np.min(Y), np.max(Y)])
        
        plt.legend()
        plt.show()
    
    
    
def main():
    q = open_q3d()
    
    close_q3d(q)
    
#%%
q = open_q3d()
### MAKE SURE TO MANUALLY MAKE um THE DEFAULT LENGTH UNIT!!!!
# %%
gate_model = define_gates()

make_gates(q,gate_model)
#%%
# make_substrate(q)
# make_dot(q,[-0.29,-0.02],0.05,layer='top')
#%%
run_sequence(q)
#%%
results = load_results()
results = interpolate_results(results)
#%%
plt.close('all')
plot_results_dict(results,max_cost_cut = 0.1,contours = True,plot_relcap = False)
#%%
Cap_props = {"MaxPass":15}
test = q.create_setup(setupname = 'MySetup' , props={'Cap':Cap_props,"AdaptiveFreq": "1MHz", "SaveFields": True, "DC": False, "AC": False})
#%%
test.__dict__
#%%
close_q3d(q)


#%%

if __name__=="__main__":
    main()
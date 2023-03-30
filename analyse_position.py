# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:23:06 2023

@author: aivlev
"""

import os
import pickle
import time

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import matplotlib.collections as mcoll
import matplotlib.path as mpath

import matplotlib as mpl

import find_position_v3 as find_pos


def calculate_cost_for_rad(results_rad_dict,goal_rel_cap = {'BL':0.45,'SSL':0.5,'BLU':0.4,'BLD':0.4},std_dict={'BL':0.05,'SSL':0.05,'BLU':0.05,'BLD':0.05}):
    cost = 0
    std = 0
    
    main_gate = 'SL'
    for gate in goal_rel_cap.keys():
        difference = results_rad_dict[gate]/results_rad_dict[main_gate]-goal_rel_cap[gate]
        cost += difference**2
        std += std_dict[gate]**2*difference**2
    
    std = (std/cost)**0.5
    cost = cost**0.5
    
    return cost, std


def load_results():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = current_dir+"\\FEM_results_2"
    os.chdir(target_dir)
    
    only_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    
    results_dict = {'top':{},'bot':{}}
    
    for idx,file in enumerate(only_files):
        rad = float(file.split('_')[1])
        rad = round(rad,3) #um
        
        layer = file.split('_')[2]
        
        with open(file, "rb") as input_file:
            results_dict[layer][str(rad)] = pickle.load(input_file)

    
    return results_dict


def interpolate_results(results_dict,N_x=100,N_y=100):
    from scipy.interpolate import RegularGridInterpolator as interpolator
    
    for layer in results_dict.keys():
        for key in results_dict[layer].keys():
            old_X = results_dict[layer][key]['X']
            old_Y = results_dict[layer][key]['Y']
            
            min_x,max_x = old_X[0,0],old_X[-1,0]
            min_y,max_y = old_Y[0,0],old_Y[0,-1]
            
            new_x = np.linspace(min_x,max_x,N_x)
            new_y = np.linspace(min_y,max_y,N_y)
            
            new_X,new_Y = np.meshgrid(new_x,new_y,indexing = 'ij')
            
            results_dict[layer][key]['X'] = new_X
            results_dict[layer][key]['Y'] = new_Y
            
            new_X = np.reshape(new_X,np.shape(new_X)+(1,))
            new_Y = np.reshape(new_Y,np.shape(new_Y)+(1,))
            points = np.concatenate((new_X,new_Y),axis=2)
            
            method = 'cubic'
            results_dict[layer][key]['SL'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[layer][key]['SL'],method=method)(points)
            results_dict[layer][key]['BL'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[layer][key]['BL'],method=method)(points)
            results_dict[layer][key]['SSL'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[layer][key]['SSL'],method=method)(points)
            results_dict[layer][key]['BLU'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[layer][key]['BLU'],method=method)(points)
            results_dict[layer][key]['BLD'] = interpolator((old_X[:,0],old_Y[0,:]),results_dict[layer][key]['BLD'],method=method)(points)
        
    return results_dict

def plot_layout(plot_name="Expected position",image_name="design_screenshot_colored_clean.png",x_offset=0,y_offset=0):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)
    
    
    img = mpimg.imread(image_name)
    
    fig = plt.figure(plot_name)
    # x = [(-0.6-x_offset)*1000,(-0.113-x_offset)*1000]
    # y = [(-0.185-y_offset)*1000,(0.157-y_offset)*1000]
    # plt.imshow(img,extent=[x[0],x[1],y[0],y[1]])
    
    # #Turned image: 
    x = [(-0.6318-x_offset)*1000,(-0.07-x_offset)*1000]
    y = [(-0.3-y_offset)*1000,(0.275-y_offset)*1000]
    plt.imshow(img,extent=[y[1],y[0],x[0],x[1]])
    
    return fig

def plot_results_dict(results_dict,max_cost_cut = 0.1,contours = True,plot_relcap = True):
    x_offset = -0.252 #um
    y_offset = -0.014 #um
    
    
    all_goals = {}
    
    # all_goals["dot1"] = {'BL':0.45,'SSL':0.53,'BLU':0.41,'BLD':0.42} # dot1
    # all_goals["dot1"] = {'BL':0.44,'SSL':0.52,'BLU':0.45,'BLD':0.37} # dot1 
    # all_goals["dot1 displaced"] = {'BL':0.45,'SSL':0.53,'BLU':0.55,'BLD':0.3} # dot1 displaced
    # all_goals["dot2"] = {'BL':0.55,'SSL':0.73,'BLU':0.54,'BLD':0.61} # dot2
    # all_goals["dot2"] = {'BL':0.59,'SSL':0.80,'BLU':0.53,'BLD':0.53} # dot2
    
    all_goals["dot1"] = {'BL':0.44,'SSL':0.53,'BLU':0.41,'BLD':0.40} # dot1 Single Transition
    all_goals["dot2"] = {'BL':0.55,'SSL':0.76,'BLU':0.53,'BLD':0.53} # dot1 Single Transition
    
    
    
    all_deviations = {}
    # all_deviations["dot1"] = {'BL':0.021,'SSL':0.029,'BLU':0.058,'BLD':0.038}
    # all_deviations["dot2"] = {'BL':0.10,'SSL':0.061,'BLU':0.047,'BLD':0.046}
    # all_deviations["dot2"] = {'BL':0.048,'SSL':0.061,'BLU':0.037,'BLD':0.046}
    
    all_deviations["dot1"] = {'BL':0.021,'SSL':0.015,'BLU':0.017,'BLD':0.016}
    all_deviations["dot2"] = {'BL':0.014,'SSL':0.037,'BLU':0.039,'BLD':0.046}
    
    
    
    all_goal_colors = {"dot1":'blue',"dot1 displaced":'green',"dot2":'orange'}
    all_goal_labels = {"dot1":'Dot 1',"dot1 displaced":'Dot 1 (displaced)',"dot2":'Dot 2'}
    
    layer_marker_shape = {"top":"o","bot":"v"}
    layer_line_shape = {"top":"--","bot":"-"}
    layer_label = {"top":"top well","bot":"bottom well"}
    
    interdot_slopes = {'BL':-4.8,'SSL':-4.3,'BLU':-3.7,'BLD':-6.4}
    
    interdot_ratio = {}
    all_interdot = []
    for key in interdot_slopes.keys():
        tmp = (1+all_goals["dot1"][key]*interdot_slopes[key])/(1+all_goals["dot2"][key]*interdot_slopes[key])
        interdot_ratio[key] = tmp
        all_interdot.append(tmp)
    
    mean_interdot = np.mean(all_interdot)
    mean_interdot = 0.7
    # print(interdot_ratio)
    # print(mean_interdot)
    
    plot_layout(x_offset=x_offset,y_offset=y_offset)
    
    layer_best = {}
    for layer in results_dict.keys():
        
        layer_best[layer] = {}
        
        for goal_key in all_goals.keys():
            
            goal_rel_cap = all_goals[goal_key]
        
            min_idx_list = []
            min_cost_list = []
            std_min_cost_list = []
            min_xy_list = []
            min_yx_list = []
            rad_list = []
            close_to_min_list = []
            cost_list = []
            std_list = []
            
            for key in results_dict[layer].keys():
                radius = float(key)
                rad_list.append(1000*radius)
                
                X = (results_dict[layer][key]['X']-x_offset)*1000
                Y = (results_dict[layer][key]['Y']-y_offset)*1000
                
                SL_abs = results_dict[layer][key]['SL']
                BL_rel = results_dict[layer][key]['BL']/SL_abs
                SSL_rel = results_dict[layer][key]['SSL']/SL_abs
                BLU_rel = results_dict[layer][key]['BLU']/SL_abs
                BLD_rel = results_dict[layer][key]['BLD']/SL_abs
                
                
                BL_close = np.abs(BL_rel-goal_rel_cap['BL'])<all_deviations[goal_key]['BL']
                SSL_close = np.abs(SSL_rel-goal_rel_cap['SSL'])<all_deviations[goal_key]['SSL']
                BLU_close = np.abs(BLU_rel-goal_rel_cap['BLU'])<all_deviations[goal_key]['BLU']
                BLD_close = np.abs(BLD_rel-goal_rel_cap['BLD'])<all_deviations[goal_key]['BLD']
                
                
                if layer != 'top':
                    # print(key)
                    current_dot = goal_key
                    dot_idx = ['dot1','dot2'].index(current_dot)
                    complementary_dot = ['dot2','dot1'][dot_idx]
                    comp_rad = layer_best['top'][complementary_dot]['rad']
                    comp_xy_idx = layer_best['top'][complementary_dot]['xy_idx']
                    
                    SL_base = np.ndarray.flatten(results_dict['top'][str(comp_rad/1000)]['SL'])[comp_xy_idx]
                    
                    epsilon = 0.05
                    
                    SL_rel = SL_abs/SL_base
                    SL_close = np.abs(SL_rel-mean_interdot)<epsilon
                else:
                    # print(key)
                    pass
                
                
                if plot_relcap:
                    
                    if layer != 'top':                       
                        plt.figure()
                        plt.title(f'SL bottom rel_cap for {goal_key} radius {1000*radius}nm')
                        im = plt.pcolor(X,Y,SL_rel)
                        plt.colorbar(im,label='rel_cap')
                    
                    plt.figure()
                    plt.title(f'BL rel_cap for {goal_key} radius {1000*radius}nm')
                    im = plt.pcolor(X,Y,BL_rel)
                    plt.colorbar(im,label='rel_cap')
                    plt.show()
                    
                    plt.figure()
                    plt.title(f'SSL rel_cap for {goal_key} radius {1000*radius}nm')
                    im = plt.pcolor(X,Y,SSL_rel)
                    plt.colorbar(im,label='rel_cap')
                    plt.show()
                    
                    plt.figure()
                    plt.title(f'BLU rel_cap for {goal_key} radius {1000*radius}nm')
                    im = plt.pcolor(X,Y,BLU_rel)
                    plt.colorbar(im,label='rel_cap')
                    plt.show()
                    
                    plt.figure()
                    plt.title(f'BLD rel_cap for {goal_key} radius {1000*radius}nm')
                    im = plt.pcolor(X,Y,BLD_rel)
                    plt.colorbar(im,label='rel_cap')
                    plt.show()
                
                
                
                plt.figure()
                plt.title(f'Cost for {layer} {goal_key} radius {1000*radius}nm')
                cost,std_cost = calculate_cost_for_rad(results_dict[layer][key],goal_rel_cap = goal_rel_cap,std_dict=all_deviations[goal_key])
                im = plt.pcolor(X,Y,cost,vmin=0, vmax=max_cost_cut)
                plt.ylabel("y-position ")
                plt.xlabel("x-position ")
                plt.colorbar(im,label='cost')
                
                if contours:
                    plt.pcolor(X, Y, BL_close, shading='auto',alpha=0.5*BL_close,cmap=mpl.colormaps['Reds'])
                    plt.pcolor(X, Y, SSL_close, shading='auto',alpha=0.5*SSL_close,cmap=mpl.colormaps['Purples'])
                    plt.pcolor(X, Y, BLU_close, shading='auto',alpha=0.5*BLU_close,cmap=mpl.colormaps['Greens'])
                    plt.pcolor(X, Y, BLD_close, shading='auto',alpha=0.5*BLD_close,cmap=mpl.colormaps['Blues'])
                    if layer != 'top':
                        plt.pcolor(X, Y, BLD_close, shading='auto',alpha=0.5*SL_close,cmap=mpl.colormaps['Greys'])
                
                # plt.show()
                
                min_idx = np.argmin(cost)
                min_cost = np.ndarray.flatten(cost)[min_idx]
                std_min_cost = np.ndarray.flatten(std_cost)[min_idx]
                
                
                min_idx_list.append(min_idx)
                min_cost_list.append(min_cost)
                std_min_cost_list.append(std_min_cost)
                min_xy = [np.ndarray.flatten(X)[min_idx],np.ndarray.flatten(Y)[min_idx]]
                min_yx = [np.ndarray.flatten(Y)[min_idx],np.ndarray.flatten(X)[min_idx]]
                min_xy_list.append(min_xy)
                min_yx_list.append(min_yx)
                

                
                cost_list.append(cost)
                std_list.append(std_cost)
                
                
            min_idx_arr = np.array(min_idx_list)
            min_cost_arr = np.array(min_cost_list)
            std_min_cost_arr = np.array(std_min_cost_list)
            
            min_xy_arr = np.array(min_xy_list)
            min_yx_arr = np.array(min_yx_list)
            rad_arr = np.array(rad_list)
            
            best_rad_idx = np.argmin(min_cost_arr)
            
            plt.figure("Expected Radius")
            # plt.plot(rad_arr,min_cost_arr,layer_marker_shape[layer]+layer_line_shape[layer],color = all_goal_colors[goal_key],label=all_goal_labels[goal_key]+" "+layer_label[layer])
            # segments = make_segments(rad_arr, min_cost_arr)
            
            # plt.errorbar(rad_arr,min_cost_arr,std_min_cost_arr,fmt = layer_marker_shape[layer]+layer_line_shape[layer],color = all_goal_colors[goal_key],label=all_goal_labels[goal_key]+" "+layer_label[layer],capsize = 5)
            color_errorbar(rad_arr,min_cost_arr,std_min_cost_arr,layer,goal_key)
            
            plt.xlabel('Radius (nm)')
            plt.ylabel('Cost at best position')
            # plt.plot(rad_arr,min_cost_arr,layer_line_shape[layer],color = all_goal_colors[goal_key],label=all_goal_labels[goal_key]+" "+layer_label[layer])
            # tmp_rad  = np.delete(rad_arr, best_rad_idx)
            # tmp_cost  = np.delete(min_cost_arr, best_rad_idx)
            # plt.plot(tmp_rad,tmp_cost,layer_marker_shape[layer],markersize=6,color = all_goal_colors[goal_key])
            # plt.plot(rad_arr[best_rad_idx],min_cost_arr[best_rad_idx],'*',markersize=10,color = all_goal_colors[goal_key])
            plt.legend()
            # plt.show()
            
            plt.figure("Expected position")
            # plt.plot(min_xy_arr[:,0],min_xy_arr[:,1],'o-',color = all_goal_colors[goal_key],label=goal_key)
            # plt.plot(min_xy_arr[best_rad_idx,0],min_xy_arr[best_rad_idx,1],'*',markersize=12,color = all_goal_colors[goal_key])
            # plt.plot(min_yx_arr[:,0],min_yx_arr[:,1],'o-',color = all_goal_colors[goal_key],label=goal_key)
            color_lineplot(min_yx_arr[:,0],min_yx_arr[:,1],layer,goal_key,best_rad_idx)
            
            # plt.plot(min_yx_arr[best_rad_idx,0],min_yx_arr[best_rad_idx,1],'*',markersize=12,color = all_goal_colors[goal_key])
            
            
            # plt.xlim([np.min(X),np.max(X)])
            # plt.ylim([np.min(Y), np.max(Y)])
            plt.xlim([100,-100])
            plt.ylim([-100,100])
            plt.xlabel("x position (nm)")
            plt.ylabel("y position (nm)")
            
            best_cost = min_cost_arr[best_rad_idx]
            best_cost_std = std_min_cost_arr[best_rad_idx]
            

            close_to_min = np.zeros(np.shape(cost_list[0]))
            rad_keys = results_dict[layer].keys()
            for idx, rad in enumerate(rad_keys):
                total_std_from_min = (std_list[idx]**2+best_cost_std**2)**0.5
                
                close_to_min += np.abs(cost_list[idx]-best_cost)<total_std_from_min   
                close_to_min = close_to_min>0.1
            
            layer_best[layer][goal_key] = {'rad': rad_arr[best_rad_idx],'xy_idx': min_idx_arr[best_rad_idx],'close':close_to_min}
            
            plt.legend()
    plt.show()
    return layer_best
    
def plot_best_dot_on_layout(results_dict):
    layer_best = plot_results_dict(results_dict,max_cost_cut = 0.1,contours = False,plot_relcap = False)
    plt.close('all')
    
    all_goal_colors = {"dot1":'blue',"dot1 displaced":'green',"dot2":'orange'}
    all_goal_cmaps = {"dot1":'Blues',"dot1 displaced":'green',"dot2":'Oranges'}
    
    x_offset = -0.252 #um
    y_offset = -0.014 #um
    
    for layer in layer_best.keys():
    
        layout_name = "Expected position layer: "+layer
        fig = plot_layout(plot_name = layout_name,x_offset = x_offset,y_offset = y_offset)
        
        for dot in layer_best[layer].keys():
            rad = layer_best[layer][dot]['rad']
            min_idx = layer_best[layer][dot]['xy_idx']
        
            X = (results_dict[layer][str(rad/1000)]['X']-x_offset)*1000
            Y = (results_dict[layer][str(rad/1000)]['Y']-y_offset)*1000
            min_xy = (np.ndarray.flatten(X)[min_idx],np.ndarray.flatten(Y)[min_idx])
            min_yx = (np.ndarray.flatten(Y)[min_idx],np.ndarray.flatten(X)[min_idx])
            
            # plt.plot(min_xy[0],min_xy[1],'x',markersize=8,color = all_goal_colors[dot])
            # circle = plt.Circle(min_xy, rad, color=all_goal_colors[dot],alpha=0.2)
            
            plt.plot(min_yx[0],min_yx[1],'x',markersize=8,color = all_goal_colors[dot])
            
            close_to_min = layer_best[layer][dot]['close']
            plt.pcolor(Y, X, close_to_min, shading='auto',alpha=0.5*close_to_min,cmap=mpl.colormaps[all_goal_cmaps[dot]],rasterized =True)
            
            
            ax = fig.gca() 
            
            # circle = plt.Circle(min_yx, rad, color=all_goal_colors[dot],alpha=0.5)
            # ax.add_patch(circle)
            
        plt.xlim([100,-100])
        plt.ylim([-120,80])
        plt.ylabel("y-position ")
        plt.xlabel("x-position ")
            
    layer_dot_dict = {'top':'dot1','bot':'dot2'}
    
    layout_name = "Overall expected system"
    fig = plot_layout(plot_name = layout_name,x_offset = x_offset,y_offset = y_offset)
    
    for layer in layer_dot_dict.keys():
        rad = layer_best[layer][layer_dot_dict[layer]]['rad']
        min_idx = layer_best[layer][layer_dot_dict[layer]]['xy_idx']
    
        X = (results_dict[layer][str(rad/1000)]['X']-x_offset)*1000
        Y = (results_dict[layer][str(rad/1000)]['Y']-y_offset)*1000
        
        min_xy = (np.ndarray.flatten(X)[min_idx],np.ndarray.flatten(Y)[min_idx])
        #Turned
        min_yx = (np.ndarray.flatten(Y)[min_idx],np.ndarray.flatten(X)[min_idx])
        
        # plt.plot(min_xy[0],min_xy[1],'x',markersize=8,color = all_goal_colors[layer_dot_dict[layer]])
        # circle = plt.Circle(min_xy, rad, color=all_goal_colors[layer_dot_dict[layer]],alpha=0.5)
        plt.plot(min_yx[0],min_yx[1],'x',markersize=8,color = all_goal_colors[layer_dot_dict[layer]])

        close_to_min = layer_best[layer][layer_dot_dict[layer]]['close']
        plt.pcolor(Y, X, close_to_min, shading='auto',alpha=0.5*close_to_min,cmap=mpl.colormaps[all_goal_cmaps[layer_dot_dict[layer]]],rasterized =True)
            
        
        ax = fig.gca() 
        # circle = plt.Circle(min_yx, rad, color=all_goal_colors[layer_dot_dict[layer]],alpha=0.5)
        # ax.add_patch(circle)
        
        plt.xlim([100,-100])
        plt.ylim([-120,80])
    plt.ylabel("y-position ")
    plt.xlabel("x-position ")
            
    plt.savefig('simulated_position_PLACEHOLDER.pdf', transparent=True, dpi=300)
    
def rerun_sim(save_file, idx_x, idx_y):
    #To correct bug in old intermediate code
    
    rad = float(save_file.split('_')[1])
    rad = round(rad,3) #um
    
    layer = save_file.split('_')[2]
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = current_dir+"\\FEM_results_2"
    os.chdir(target_dir)
    
    with open(save_file, "rb") as input_file:
        rad_dict = pickle.load(input_file)
    
    pos_x = rad_dict['X'][idx_x,idx_y]
    pos_y = rad_dict['Y'][idx_x,idx_y]
    
    SL_array = rad_dict['SL']
    BL_array = rad_dict['BL']
    SSL_array = rad_dict['SSL']
    BLU_array = rad_dict['BLU']
    BLD_array = rad_dict['BLD']
    
    q = find_pos.open_q3d()
    gate_model = find_pos.define_gates()

    find_pos.make_gates(q,gate_model)
    
    find_pos.make_substrate(q)
    find_pos.make_dot(q,[pos_x,pos_y],rad,layer=layer)
    result = find_pos.analyse(q,SaveFields=False)
    find_pos.clear_analysis(q)
    gate_list,rel_cap_list,abs_cap_list = find_pos.post_process_result(result)
    
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

    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = current_dir+"\\FEM_results_2"
    name = f"dotRadius_{rad}_{layer}_relCapList_{str(int(time.time()))}.pkl"
    
    os.chdir(target_dir)
    
    with open(name, 'wb') as f:
        pickle.dump(rad_dict, f)

def fake_legend(markerstyle,linestyle,color,label):
    plt.plot(100,100,markerstyle+linestyle,color = color,label=label)

def color_lineplot(x,y,layer,dot,best_rad_idx):
    layer_marker_shape = {"top":"o","bot":"v"}
    layer_line_shape = {"top":"--","bot":"-"}
    layer_label = {"top":"Top well","bot":"Bottom well"}
    dot_colormap = {"dot1":'Blues',"dot2":'Oranges'}
    dot_labels = {"dot1":'dot 1',"dot2":'dot 2'}
    
    z_marker = np.linspace(0.3, 0.7, len(x))
    
    for i in range(len(x)):
        plt.errorbar(x[i],y[i],fmt = layer_marker_shape[layer],color = plt.get_cmap(dot_colormap[dot])(z_marker[i]),capsize = 5)
    
    # plt.plot(x[best_rad_idx],y[best_rad_idx],'*',markersize=12,color = plt.get_cmap(dot_colormap[dot])(z_marker[best_rad_idx]))
    # fake_legend(markerstyle = layer_marker_shape[layer],linestyle = layer_line_shape[layer],color=plt.get_cmap(dot_colormap[dot])(0.6),label = layer_label[layer]+dot_labels[dot])
    
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x_line, y_line = verts[:, 0], verts[:, 1]
    z_line = np.linspace(0.3, 0.7, len(x_line))
    
    colorline(x_line, y_line, z_line, cmap=plt.get_cmap(dot_colormap[dot]), linewidth=2,linestyle=layer_line_shape[layer])
    

def color_errorbar(x,y,y_err,layer,dot):
    layer_marker_shape = {"top":"o","bot":"v"}
    layer_line_shape = {"top":"--","bot":"-"}
    layer_label = {"top":"Top well" ,"bot":"Bottom well "}
    dot_colormap = {"dot1":'Blues',"dot2":'Oranges'}
    dot_labels = {"dot1":'dot 1',"dot2":'dot 2'}
    
    z_marker = np.linspace(0.3, 0.7, len(x))
    
    for i in range(len(x)):
        plt.errorbar(x[i],y[i],y_err[i],fmt = layer_marker_shape[layer],color = plt.get_cmap(dot_colormap[dot])(z_marker[i]),capsize = 5)
    
    fake_legend(markerstyle = layer_marker_shape[layer],linestyle = layer_line_shape[layer],color=plt.get_cmap(dot_colormap[dot])(0.6),label = layer_label[layer]+dot_labels[dot])
    
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x_line, y_line = verts[:, 0], verts[:, 1]
    z_line = np.linspace(0.3, 0.7, len(x_line))
    
    colorline(x_line, y_line, z_line, cmap=plt.get_cmap(dot_colormap[dot]), linewidth=2,linestyle=layer_line_shape[layer])
    
    plt.xlim(15,105)
    plt.ylim(0,0.7)
    

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0,linestyle='-'):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha,linestyle=linestyle)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x,y):
    # To make gradually changing color
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

    
#%%    
def main():
    results = load_results()
    results = interpolate_results(results)
    
    plt.close('all')
    plot_results_dict(results,max_cost_cut = 1,contours = False,plot_relcap = False)
    
    plt.close('all')
    plot_best_dot_on_layout(results)
    
if __name__=="__main__":
    main()


#%%
results = load_results()
results = interpolate_results(results)

#%%

plt.close('all')
plot_results_dict(results,max_cost_cut = 1,contours = True,plot_relcap = False)

#%%
plt.close('all')
plot_best_dot_on_layout(results)
#%%
results = load_results()
plt.close('all')
plot_results_dict(results,max_cost_cut = 1,contours = False,plot_relcap = True)

#%%
rerun_sim(save_file="dotRadius_0.1_top_relCapList_1679290661.pkl",idx_x=1,idx_y=1)
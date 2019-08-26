import plotsAnalysis
import flag_reader
import pandas as pd
import time_recorder
TK = time_recorder.time_keeper()
TK.record(5)
#flags = flag_reader.read_flag()
#flag_reader.write_flags_and_BVE(flags, 0.01)
#df = pd.read_csv('parameters.txt', index_col = 0);
#print(df)
#print(df['best_validation_loss'])
#plotsAnalysis.HeatMapBVL(plot_x_name = '#backward_layer', plot_y_name='#con_layer',
#                        title = 'Loss heatmap for # of backward and conv layers',HeatMap_dir = "../swipe_NC_1000_analysis",
#                        feature_1_name = 'backward_fc_filters', feature_2_name = 'conv1d_filters')
#plotsAnalysis.HeatMapBVL(plot_x_name = '#backward_layer', plot_y_name='bv_loss',
#                        title = 'HeatMap.png',HeatMap_dir = "../swipe_NC_1000_analysis",
#                        feature_1_name = 'backward_fc_filters', feature_2_name = None)

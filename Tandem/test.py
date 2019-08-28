import plotsAnalysis
import flag_reader
import pandas as pd
import time_recorder
import plotsAnalysis

#plotsAnalysis.PlotPossibleGeoSpace("Possible geo space for tandem structure","data/", compare_original = False)

#TK = time_recorder.time_keeper()
#TK.record(5)
#flags = flag_reader.read_flag()
#flag_reader.write_flags_and_BVE(flags, 0.01)
#df = pd.read_csv('parameters.txt', index_col = 0);
#print(df)
#print(df['best_validation_loss'])
plotsAnalysis.HeatMapBVL(plot_x_name = '#conv_channel_list', plot_y_name='#con1d_filters',
                        title = 'Loss heatmap for # of conv1d filters',HeatMap_dir = "../swipe_conv_analysis",
                        feature_1_name = 'conv_channel_list', feature_2_name = 'conv1d_filters',condense_tuple2len = False)
#plotsAnalysis.HeatMapBVL(plot_x_name = '#conv_channel_list', plot_y_name='bv_loss',
#                        title = 'HeatMap.png',HeatMap_dir = "../swipe_conv",
#                        feature_1_name = 'conv_channel_list', feature_2_name = None,condense_tuple2len =False)

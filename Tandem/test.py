import plotsAnalysis
import flag_reader
import pandas as pd
#flags = flag_reader.read_flag()
#flag_reader.write_flags_and_BVE(flags, 0.01)
#df = pd.read_csv('parameters.txt', index_col = 0);
#print(df)
#print(df['best_validation_loss'])
plotsAnalysis.HeatMapBVL('#backward_layer',None,'HeatMap.png',HeatMap_dir = "../swipe_NC_1000_analysis",feature_1_name = 'backward_fc_filters')

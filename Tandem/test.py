import plotsAnalysis
import flag_reader
import pandas as pd
flags = flag_reader.read_flag()
flag_reader.write_flags_and_BVE(flags, 0.01)
df = pd.read_csv('parameters.txt', index_col = 0);
print(df)
#print(df['best_validation_loss'])
#plotsAnalysis.HeatMapBVL('backward_fc_filters',None,'HeatMap.png',HeatMap_dir = "HeatMap",feature_1_name = '#backward_layer')
